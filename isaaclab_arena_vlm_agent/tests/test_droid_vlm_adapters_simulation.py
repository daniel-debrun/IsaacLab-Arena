# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify DROID VLM action and observation contracts against live simulation."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_droid_absolute_pose_adapters(simulation_app):
    import math
    import torch

    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.embodiments.common.command_type import CommandType
    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_vlm_agent.action_adapter import StepActionAdapter
    from isaaclab_arena_vlm_agent.commands import Command
    from isaaclab_arena_vlm_agent.observation_adapter import StateObservationAdapter
    from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator

    # Use a translated, rotated root and separate environment origins to expose world/root confusion.
    embodiment = DroidDifferentialIKEmbodiment(
        initial_pose=Pose(
            position_xyz=(0.7, -0.4, 1.35),
            rotation_xyzw=(0.0, 0.0, math.sin(0.3), math.cos(0.3)),
        )
    )

    def absolute_pose_actions(cfg):
        cfg.actions.arm_action.controller.use_relative_mode = False
        cfg.actions.arm_action.scale = 1.0
        return cfg

    arena_env = IsaacLabArenaEnvironment(
        name="droid_vlm_adapter_test",
        scene=Scene(),
        embodiment=embodiment,
        env_cfg_callback=absolute_pose_actions,
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        observation_adapter = StateObservationAdapter(embodiment)
        action_adapter = StepActionAdapter(embodiment, CommandType.ABS_EE_POSE)
        assert embodiment.get_binary_action_values(env) == {7: (0.0, 1.0)}
        assert action_adapter.is_robot_action_space_supported(env)
        states = [observation_adapter.extract_robot_state(env, obs, env_id) for env_id in range(base.num_envs)]
        actions = []
        for env_id, state in enumerate(states):
            pose = state["command_body_poses"]["arm_action"]
            response = Command(values=pose["position_root_m"] + pose["quaternion_root_xyzw"] + [0.0])
            command = action_adapter.decode_command(response, base.arena_world, env_id)
            actions.append(action_adapter.command_to_action(env, command))
        actions = torch.stack(actions)
        assert actions.shape == (2, 8)
        T_R_E = actions[:, :7]
        torch.testing.assert_close(embodiment.get_measured_command_body_pose_in_root(env), T_R_E)
        T_W_R = base.arena_world.get_pose_w("robot")
        t_W_E, q_W_E = combine_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], T_R_E[:, :3], T_R_E[:, 3:])
        torch.testing.assert_close(
            torch.cat((t_W_E, q_W_E), dim=-1),
            base.arena_world.get_body_pose_w("robot", "base_link"),
            atol=1e-5,
            rtol=1e-5,
        )
        # Verify raw action packing reaches the controller with no scale or offset conversion.
        base.action_manager.process_action(actions)
        arm = base.action_manager.get_term("arm_action")
        torch.testing.assert_close(arm.processed_actions, actions[:, :7])
        gripper = base.action_manager.get_term("gripper_action")
        assert torch.all(gripper.processed_actions == 0)
        actions[:, 7] = 1.0
        base.action_manager.process_action(actions)
        assert torch.all(gripper.processed_actions > 0)
        actions[:, 7] = 0.0
        for _ in range(5):
            obs, _, _, _, _ = env.step(actions)
        measured_states = [observation_adapter.extract_robot_state(env, obs, env_id) for env_id in range(base.num_envs)]
        for env_id, state in enumerate(measured_states):
            assert abs(state["policy"]["gripper_pos"][0] - obs["policy"]["gripper_pos"][env_id, 0].item()) < 1e-6

        generator = PoseTargetMotionGenerator(embodiment)
        for env_id, state in enumerate(measured_states):
            pose = state["command_body_poses"]["arm_action"]
            response = Command(values=pose["position_root_m"] + pose["quaternion_root_xyzw"] + [0.0])
            goal = action_adapter.decode_command(response, base.arena_world, env_id)
            assert generator.is_goal_reached(goal, env, env_id) == (
                abs(state["policy"]["gripper_pos"][0]) <= generator.gripper_tolerance
            )
            goal[0] += 0.1
            assert not generator.is_goal_reached(goal, env, env_id)

    finally:
        env.close()
    return True


def test_droid_absolute_pose_adapters():
    assert run_function_with_persistent_simulation_app(_test_droid_absolute_pose_adapters)


def _test_droid_joint_position_adapters(simulation_app):
    import torch

    from isaaclab_arena.embodiments.common.command_type import CommandType
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena_vlm_agent.action_adapter import StepActionAdapter
    from isaaclab_arena_vlm_agent.commands import Command

    embodiment = DroidAbsoluteJointPositionEmbodiment()
    arena_env = IsaacLabArenaEnvironment(name="droid_vlm_joint_test", scene=Scene(), embodiment=embodiment)
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        adapter = StepActionAdapter(embodiment, CommandType.ABS_JOINT_POSITION)
        assert adapter.is_robot_action_space_supported(env)
        joint_names = tuple(f"panda_joint{i}" for i in range(1, 8))
        targets = base.arena_world.get_joint_positions("robot", joint_names)
        actions = []
        for env_id, target in enumerate(targets):
            command = adapter.decode_command(Command(values=target.tolist() + [0.0]), base.arena_world, env_id)
            actions.append(adapter.command_to_action(env, command))
        actions = torch.stack(actions)
        torch.testing.assert_close(actions[:, :7], targets)
        base.action_manager.process_action(actions)
        torch.testing.assert_close(base.action_manager.get_term("arm_action").processed_actions, targets)
        assert torch.all(base.action_manager.get_term("gripper_action").processed_actions == 0)
        env.step(actions)
    finally:
        env.close()
    return True


def test_droid_joint_position_adapters():
    assert run_function_with_persistent_simulation_app(_test_droid_joint_position_adapters)
