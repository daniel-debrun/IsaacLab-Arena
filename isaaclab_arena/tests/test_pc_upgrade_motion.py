# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Continuity checks for the assisted PC-upgrade trajectory."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_minimum_jerk_trajectory(simulation_app):
    import pytest

    from isaaclab_arena_examples.pc_upgrade.motion import minimum_jerk

    duration = 4.0
    assert minimum_jerk(-1.0, duration) == (0.0, 0.0, 0.0)
    assert minimum_jerk(duration + 1.0, duration) == (1.0, 0.0, 0.0)
    assert minimum_jerk(0.0, duration) == (0.0, 0.0, 0.0)
    assert minimum_jerk(duration, duration) == (1.0, 0.0, 0.0)
    dt = 1.0 / 120.0
    previous = 0.0
    for index in range(1, 481):
        fraction, velocity, acceleration = minimum_jerk(index * dt, duration)
        assert fraction >= previous
        assert fraction - previous < 0.004
        assert velocity >= -1e-12
        previous = fraction
    # Numerical derivatives verify physical units when duration changes.
    for time in (0.2, 1.0, 2.7, 3.8):
        before = minimum_jerk(time - 1e-5, duration)
        position, velocity, acceleration = minimum_jerk(time, duration)
        after = minimum_jerk(time + 1e-5, duration)
        assert velocity == pytest.approx((after[0] - before[0]) / 2e-5, abs=1e-8)
        assert acceleration == pytest.approx((after[1] - before[1]) / 2e-5, abs=1e-8)
    return True


def test_minimum_jerk_trajectory():
    assert run_function_with_persistent_simulation_app(_test_minimum_jerk_trajectory)


def _test_assisted_motion_partial_reset(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pc_upgrade_environment import PCUpgradeEnvironment, PCUpgradeEnvironmentCfg
    from isaaclab_arena_examples.pc_upgrade.motion import AssistedMotionCfg

    builder = ArenaEnvBuilder(
        PCUpgradeEnvironment().build(PCUpgradeEnvironmentCfg()),
        ArenaEnvBuilderCfg(num_envs=2, env_spacing=3.0, solve_relations=False),
    )
    cfg, kwargs = builder.compose_manager_cfg()
    cfg.actions.assisted_motion = AssistedMotionCfg()
    cfg.scene.pc_chassis.actuators["hinge"].stiffness = 30.0
    cfg.scene.pc_chassis.actuators["hinge"].damping = 8.0
    env = builder.make_registered(cfg, kwargs)
    try:
        env.reset()
        base = env.unwrapped
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, -1] = 1.0
        motion = base.action_manager.get_term("assisted_motion")
        gpu = base.scene["rtx_5090"]
        target = gpu.data.root_pos_w[:].clone()
        target[:, 2] += 0.25
        motion.move("rtx_5090", target, gpu.data.root_quat_w[:].clone(), duration=1.0)
        motion.move_panel(-1.0, duration=1.0)
        for _ in range(25):
            env.step(action)
        assert torch.all(torch.norm(gpu.data.root_pos_w[:] - target, dim=-1) < 0.005)
        base._reset_idx(torch.tensor([0], device=base.device))
        for _ in range(20):
            env.step(action)
        # Resetting clone zero must not release clone one's airborne grasp or door hold.
        assert torch.norm(gpu.data.root_pos_w[1] - target[1]) < 0.005
        assert gpu.data.root_pos_w[0, 2] < target[0, 2] - 0.20
        assert abs(base.scene["pc_chassis"].data.joint_pos[1, 0] + 1.0) < 0.02
        assert abs(base.scene["pc_chassis"].data.joint_pos[0, 0]) < 0.02
        motion.release()
        for _ in range(20):
            env.step(action)
        assert gpu.data.root_pos_w[1, 2] < target[1, 2] - 0.20
        return True
    finally:
        env.close()


def test_assisted_motion_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_assisted_motion_partial_reset)
