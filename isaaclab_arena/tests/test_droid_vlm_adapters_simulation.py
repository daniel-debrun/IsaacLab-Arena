# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify DROID VLM action and observation contracts against live simulation."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_droid_absolute_pose_adapters(simulation_app):
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter
    from isaaclab_arena_vlm_agent.commands import Command, CommandType
    from isaaclab_arena_vlm_agent.droid_motion_generator import DroidPoseMotionGenerator
    from isaaclab_arena_vlm_agent.droid_observation_adapter import DroidObservationAdapter

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
        observation_adapter = DroidObservationAdapter()
        action_adapter = AgentActionAdapter(embodiment, CommandType.EE_POSE)
        assert action_adapter.embodiment is embodiment
        assert action_adapter.is_robot_action_space_supported(env)
        assert not AgentActionAdapter(embodiment, CommandType.JOINT_POSITION).is_robot_action_space_supported(env)
        for scene_key, body_name in (("other_robot", "base_link"), ("robot", "panda_link7")):
            other_embodiment = SimpleNamespace(
                get_scene_key=lambda: scene_key,
                get_command_body_name=lambda: body_name,
            )
            assert not AgentActionAdapter(other_embodiment, CommandType.EE_POSE).is_robot_action_space_supported(env)
        states = [observation_adapter.extract_robot_state(env, obs, env_id) for env_id in range(base.num_envs)]
        actions = []
        for env_id, state in enumerate(states):
            response = Command(
                arm_target=state["position_root_m"] + state["quaternion_root_xyzw"],
                gripper_closed=False,
            )
            command = action_adapter.decode_command(response, base.arena_world, env_id)
            actions.append(action_adapter.command_to_action(env, command))
        actions = torch.stack(actions)
        assert actions.shape == (2, 8)
        T_R_E = actions[:, :7]
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
        assert len(measured_states) == 2
        for env_id, state in enumerate(measured_states):
            assert abs(state["gripper_closed_fraction"] - obs["policy"]["gripper_pos"][env_id, 0].item()) < 1e-6

        generator = DroidPoseMotionGenerator()
        for env_id, state in enumerate(measured_states):
            response = Command(
                arm_target=state["position_root_m"] + state["quaternion_root_xyzw"],
                gripper_closed=False,
            )
            goal = action_adapter.decode_command(response, base.arena_world, env_id)
            assert generator.goal_reached(goal, base.arena_world, env_id) == (
                abs(state["gripper_closed_fraction"]) <= generator.gripper_tolerance
            )
            goal[0] += 0.1
            assert not generator.goal_reached(goal, base.arena_world, env_id)

        # Reject configurations whose native action semantics differ from the response contract.
        for field, value in (("scale", 0.5), ("body_name", "panda_link7"), ("body_offset", object()), ("clip", {})):
            original = getattr(arm.cfg, field)
            setattr(arm.cfg, field, value)
            assert not action_adapter.is_robot_action_space_supported(env), field
            setattr(arm.cfg, field, original)
        arm.cfg.controller.use_relative_mode = True
        assert not action_adapter.is_robot_action_space_supported(env)
        arm.cfg.controller.use_relative_mode = False
        assert action_adapter.is_robot_action_space_supported(env)
    finally:
        env.close()
    return True


def test_droid_absolute_pose_adapters():
    assert run_function_with_persistent_simulation_app(_test_droid_absolute_pose_adapters)


def _test_droid_joint_position_adapters(simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import Mock

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter
    from isaaclab_arena_vlm_agent.commands import CommandType
    from isaaclab_arena_vlm_agent.droid_observation_adapter import DroidObservationAdapter
    from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMChunkPolicy, VLMPolicyCfg, VLMStepPolicy

    embodiment = DroidAbsoluteJointPositionEmbodiment()
    arena_env = IsaacLabArenaEnvironment(name="droid_vlm_joint_test", scene=Scene(), embodiment=embodiment)
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        assert not AgentActionAdapter(embodiment, CommandType.EE_POSE).is_robot_action_space_supported(env)
        arm = base.action_manager.get_term("arm_action")
        joint_names = tuple(f"panda_joint{i}" for i in range(1, 8))
        targets = base.arena_world.get_joint_positions("robot", joint_names)[:1].expand(2, -1)
        target = {"arm_target": targets[0].tolist(), "gripper_closed": False}
        invalid = {"arm_target": [0.1, 0.2], "gripper_closed": False}
        nonfinite = {"arm_target": [float("nan")] * 7, "gripper_closed": False}
        observation_adapter = DroidObservationAdapter()
        observation_adapter.build_request_content = Mock(return_value=[])
        for policy_type, adapter in (
            (VLMStepPolicy, AgentActionAdapter(embodiment, CommandType.JOINT_POSITION)),
            (VLMChunkPolicy, ChunkActionAdapter(embodiment, CommandType.JOINT_POSITION, chunk_size=2)),
        ):
            assert adapter.is_robot_action_space_supported(env)
            responses = [invalid, nonfinite, target, target]
            if policy_type is VLMChunkPolicy:
                # A ragged chunk and a nonfinite chunk must be corrected before any command is executed.
                responses = [{"commands": [row, target]} for row in responses]
            backend = SimpleNamespace(infer=Mock(side_effect=responses))
            policy = policy_type(
                VLMPolicyCfg(system_prompt="Return absolute joint positions in arm controller order."),
                inference_backend=backend,
                observation_adapter=observation_adapter,
                action_adapter=adapter,
            )
            policy.set_task_description("Hold the current joints")
            actions = policy.get_action(env, {})
            assert backend.infer.call_count == 4
            torch.testing.assert_close(actions[:, :7], targets)
            base.action_manager.process_action(actions)
            torch.testing.assert_close(arm.processed_actions, targets)
            assert torch.all(base.action_manager.get_term("gripper_action").processed_actions == 0)
            if policy_type is VLMChunkPolicy:
                torch.testing.assert_close(policy.get_action(env, {}), actions)
                assert backend.infer.call_count == 4
            env.step(actions)

        # Absolute positions must not be sent through scaled, offset, or clipped controllers.
        for field, value in (("scale", 0.5), ("offset", 0.1), ("use_default_offset", True), ("clip", {})):
            original = getattr(arm.cfg, field)
            setattr(arm.cfg, field, value)
            assert not adapter.is_robot_action_space_supported(env), field
            setattr(arm.cfg, field, original)
        assert adapter.is_robot_action_space_supported(env)
    finally:
        env.close()
    return True


def test_droid_joint_position_adapters():
    assert run_function_with_persistent_simulation_app(_test_droid_joint_position_adapters)
