# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live partial-reset regression for independent coffee demo grasps and joint drives."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_coffee_assistance_partial_reset(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.coffee_disposal_environment import (
        CoffeeDisposalEnvironment,
        CoffeeDisposalEnvironmentCfg,
    )
    from isaaclab_arena_examples.coffee_disposal.motion import CoffeeMotionCfg, configure_demo_drives

    builder = ArenaEnvBuilder(
        CoffeeDisposalEnvironment().build(CoffeeDisposalEnvironmentCfg()),
        ArenaEnvBuilderCfg(num_envs=2, env_spacing=4.0, solve_relations=False),
    )
    cfg, kwargs = builder.compose_manager_cfg()
    cfg.actions.coffee_motion = CoffeeMotionCfg()
    configure_demo_drives(cfg)
    env = builder.make_registered(cfg, kwargs)
    try:
        env.reset()
        base = env.unwrapped
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, -1] = 1.0
        motion = base.action_manager.get_term("coffee_motion")
        motion.move_joint("waste_cabinet", "drawer_slide", -0.55, 2.0)
        for _ in range(50):
            env.step(action)
        lid = base.scene["bin_lid"]
        target = lid.data.root_pos_w[:].clone()
        target[:, 2] += 0.12
        motion.move("bin_lid", target, lid.data.root_quat_w[:].clone(), 1.0)
        for _ in range(25):
            env.step(action)
        assert torch.all(torch.linalg.vector_norm(lid.data.root_pos_w[:] - target, dim=-1) < 0.01)
        base._reset_idx(torch.tensor([0], device=base.device))
        for _ in range(20):
            env.step(action)
        assert not base.coffee_grasps["bin_lid"][0] and base.coffee_grasps["bin_lid"][1]
        assert torch.linalg.vector_norm(lid.data.root_pos_w[1] - target[1]) < 0.01
        drawer = base.scene["waste_cabinet"]
        index = drawer.find_joints("drawer_slide")[0][0]
        assert abs(drawer.data.joint_pos[0, index]) < 0.02
        assert abs(drawer.data.joint_pos[1, index] + 0.55) < 0.02
        motion.release()
        for _ in range(20):
            env.step(action)
        assert not base.coffee_grasps["bin_lid"].any()
        assert lid.data.root_pos_w[1, 2] < target[1, 2] - 0.08
        return True
    finally:
        env.close()


def test_coffee_assistance_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_coffee_assistance_partial_reset)


def _test_raised_dispenser_holds_without_assistance(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.coffee_disposal_environment import (
        CoffeeDisposalEnvironment,
        CoffeeDisposalEnvironmentCfg,
    )

    builder = ArenaEnvBuilder(
        CoffeeDisposalEnvironment().build(CoffeeDisposalEnvironmentCfg()),
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
    )
    env = builder.make_registered()
    try:
        env.reset()
        base = env.unwrapped
        machine = base.scene["coffee_machine"]
        positions = machine.data.joint_pos[:].clone()
        index = machine.find_joints("spout_slide")[0][0]
        positions[:, index] = 0.065
        machine.write_joint_state_to_sim(positions, torch.zeros_like(positions))
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, -1] = 1.0
        for _ in range(30):
            env.step(action)
        assert torch.all(
            (machine.data.joint_pos[:, index] - 0.065).abs() < 0.005
        ), "Released dispenser must hold its raised setting"
        assert "coffee_motion" not in base.action_manager.active_terms
        return True
    finally:
        env.close()


def test_raised_dispenser_holds_without_assistance():
    assert run_function_with_persistent_simulation_app(_test_raised_dispenser_holds_without_assistance)


def _test_coffee_terminal_records_live_owner_before_reset(simulation_app):
    import copy
    import json
    import torch
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena_examples.coffee_disposal.demo import Demo, record_demo_terminal

    # The recorder configuration is copied before Demo.__init__ assigns its base.
    partial_demo = Demo.__new__(Demo)
    partial_demo.args = SimpleNamespace(capture=False)
    partial_demo.video = None
    partial_demo.evidence = []
    old_config = copy.deepcopy(EpisodeRecorderTermCfg(func=partial_demo.terminal))
    new_config = copy.deepcopy(EpisodeRecorderTermCfg(func=record_demo_terminal))
    partial_demo.base = SimpleNamespace()
    terminal_env = SimpleNamespace(progress_tracker=SimpleNamespace(is_complete=lambda: torch.tensor([True])))
    assert old_config.func.__self__ is not partial_demo
    with pytest.raises(AttributeError, match="base"):
        old_config.func(terminal_env, 0)
    assert new_config.func is record_demo_terminal
    assert new_config.func(terminal_env, 0) == {}, "Initial reset may precede live-owner registration"

    with TemporaryDirectory(prefix="coffee-terminal-") as directory:
        args = SimpleNamespace(
            output=Path(directory),
            enable_cameras=False,
            video=True,
            capture=False,
            num_envs=2,
            device="cuda:0",
            real_time=False,
        )
        # Exercise video recorder composition without creating cameras or an encoder.
        with patch("isaaclab_arena_examples.coffee_disposal.video.TaskVideo", return_value=None):
            demo = Demo(args)
        try:
            base = demo.base
            assert base.coffee_demo is demo
            assert "progress_tracking" in base.recorder_manager.active_terms
            assert "video" in base.recorder_manager.active_terms
            # Distinct terminal poses prove the snapshot precedes resetting assets.
            ground = base.scene["grounds_0"]
            terminal_state = ground.data.root_state_w[:].clone()
            terminal_state[:, 0] += terminal_state.new_tensor([0.04, 0.08])
            ground.write_root_state_to_sim(terminal_state)
            terminal_position = (ground.data.root_pos_w[:] - base.scene.env_origins).tolist()
            cabinet = base.scene["waste_cabinet"]
            joints = cabinet.data.joint_pos[:].clone()
            drawer_index = cabinet.find_joints("drawer_slide")[0][0]
            joints[:, drawer_index] = -0.4
            cabinet.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            tracker = base.progress_tracker
            tracker._task_success.fill_(True)
            for runner in tracker.runners:
                for name in runner.sequence_complete:
                    runner.sequence_complete[name].fill_(True)
                    runner.sequence_score[name].fill_(1.0)
                    runner.current_predicate_index[name].fill_(len(runner.predicate_chains[name]))

            shots = []
            holds = []

            def record_shot(view, title):
                assert base.coffee_demo is demo
                assert tracker.is_complete().all(), "Terminal video must run before tracker reset"
                assert torch.allclose(cabinet.data.joint_pos[:, drawer_index], joints[:, drawer_index])
                shots.append((view, title))

            demo.video = SimpleNamespace(active=True, shot=record_shot, hold=holds.append)
            base.recorder_manager.record_post_step()
            base._reset_idx(torch.tensor([0], device=base.device))
            assert len(shots) == 1 and shots[0][0] == "Overview"
            assert holds == [4.0]
            assert not demo.video.active
            assert len(demo.evidence) == 1, "The live owner must receive the terminal evidence"
            terminal = demo.evidence[0]
            assert terminal["stage"] == "11_success"
            assert all(terminal["completed"][0])
            assert terminal["grounds_0"] == terminal_position
            assert terminal["waste_cabinet_joints"] == joints.tolist()
            assert tracker.is_complete().tolist() == [False, True]
            assert cabinet.data.joint_pos[0, drawer_index].abs() < 1e-6
            assert not torch.allclose(ground.data.root_pos_w[0], terminal_state[0, :3])
            assert torch.allclose(ground.data.root_pos_w[1], terminal_state[1, :3])
            # The later environment must retain its own final state even if extras
            # are unavailable or stale at a manually invoked reset boundary.
            base.extras.pop("progress_tracking", None)
            base._reset_idx(torch.tensor([1], device=base.device))
            assert not tracker.is_complete().any()
            assert cabinet.data.joint_pos[:, drawer_index].abs().max() < 1e-6
            assert len(shots) == 1 and len(demo.evidence) == 1, "Only env zero produces the terminal video shot"
            records = [json.loads(line) for line in (args.output / "episodes.jsonl").read_text().splitlines()]
            assert len(records) == 2, "Staggered terminal boundaries must persist one record per environment"
            assert records[0]["progress"]["all_complete"]
            for env_id, record in enumerate(records):
                assert record["env_id"] == env_id
                final = record["coffee_terminal_state"]
                assert all(final["completed"])
                assert final["objects"]["grounds_0"][:3] == terminal_position[env_id]
                assert final["joints"]["waste_cabinet"] == joints[env_id].tolist()
            return True
        finally:
            demo.env.close()


def test_coffee_terminal_records_live_owner_before_reset():
    assert run_function_with_persistent_simulation_app(_test_coffee_terminal_records_live_owner_before_reset)
