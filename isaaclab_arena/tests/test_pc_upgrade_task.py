# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task history and partial reset tests using actual predicates and Arena progress."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_pc_upgrade_order_and_reset(simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena_environments.pc_upgrade.layout import BIN_POSITION, CARD_POSITIONS, PC_POSITION, SLOT_IN_PC
    from isaaclab_arena_environments.pc_upgrade.task import PCUpgradeTask

    slot = torch.tensor(PC_POSITION) + torch.tensor(SLOT_IN_PC)

    # Explicit scene data exercises real predicates without manipulating a live robot.
    def obj(position):
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([position, position]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
                root_vel_w=torch.zeros(2, 6),
                joint_pos=torch.zeros(2, 1),
                joint_vel=torch.zeros(2, 1),
            )
        )

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    scene = Scene(
        pc_chassis=obj(PC_POSITION),
        rtx_5090=obj(CARD_POSITIONS["rtx_5090"]),
        vortex_vx880=obj(CARD_POSITIONS["vortex_vx880"]),
        cinder_c700=obj(CARD_POSITIONS["cinder_c700"]),
    )
    scene["ee_frame"] = SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.zeros(2, 1, 3)))
    env = SimpleNamespace(scene=scene, num_envs=2, device="cpu")
    config = PCUpgradeTask().get_termination_cfg()
    tracker = ProgressTracker(config.success, 2, "cpu", env=env, subtasks_are_sequential=True)

    step_index = torch.zeros(2, dtype=torch.long)

    def advance(count=20):
        for _ in range(count):
            tracker.step(env, step_index=step_index)
            step_index.add_(1)

    # Pre-positioning all final objects must never skip opening and lifting.
    scene["rtx_5090"].data.root_pos_w[:] = slot
    scene["vortex_vx880"].data.root_pos_w[:] = torch.tensor(BIN_POSITION) + torch.tensor([0.0, 0.0, 0.05])
    scene["cinder_c700"].data.root_pos_w[:] = torch.tensor(BIN_POSITION) + torch.tensor([0.0, 0.0, 0.10])
    advance()
    assert not tracker.get_subtask_completion().any()
    # Normal first stages, with a partial reset while environment 1 is underway.
    scene["rtx_5090"].data.root_pos_w[:] = torch.tensor([0.42, -0.25, 1.0])
    scene["pc_chassis"].data.joint_pos[:] = -1.8
    advance(5)
    assert tracker.get_subtask_completion()[:, 0].all()
    tracker.reset(torch.tensor([0]))
    assert not tracker.get_subtask_completion()[0].any()
    assert tracker.get_subtask_completion()[1, 0]
    advance(10)
    assert not tracker.get_subtask_completion()[:, 1].any(), "Pre-held GPU cannot count as a new pickup"
    scene["rtx_5090"].data.root_pos_w[:] = torch.tensor([0.42, -0.25, 0.775])
    advance(2)
    scene["rtx_5090"].data.root_pos_w[:] = torch.tensor([0.42, -0.25, 1.0])
    advance(10)
    # Arriving at the final pose without entering through the side is insufficient.
    scene["rtx_5090"].data.root_pos_w[:] = slot
    scene["rtx_5090"].data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    advance(30)
    assert not tracker.get_subtask_completion()[:, 2].any(), "Insertion must observe the socket approach"
    scene["rtx_5090"].data.root_pos_w[:] = slot + torch.tensor([0.0, -0.20, 0.0])
    advance(5)
    scene["rtx_5090"].data.root_pos_w[:] = slot
    # Face-up cards must not satisfy the connector orientation check.
    scene["rtx_5090"].data.root_quat_w[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    advance(20)
    assert not tracker.get_subtask_completion()[:, 2].any()
    scene["rtx_5090"].data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    advance(30)
    scene["pc_chassis"].data.joint_pos[:] = 0.0
    advance(40)
    assert tracker.get_subtask_completion()[:, :4].all()
    # Cards recycled before the close stage may not count as post-close disposal.
    assert not tracker.is_complete().any()
    assert not tracker.get_subtask_completion()[:, 4].any()
    scene["vortex_vx880"].data.root_pos_w[:] = torch.tensor([0.15, -0.49, 0.775])
    advance(2)
    scene["vortex_vx880"].data.root_pos_w[:] = torch.tensor([0.15, -0.49, 1.2])
    advance(3)
    scene["vortex_vx880"].data.root_pos_w[:] = torch.tensor(BIN_POSITION) + torch.tensor([0.0, 0.0, 0.05])
    advance(10)
    assert tracker.get_subtask_completion()[:, 4].all()
    scene["cinder_c700"].data.root_pos_w[:] = torch.tensor([0.52, -0.49, 0.775])
    advance(3)
    scene["cinder_c700"].data.root_pos_w[:] = torch.tensor([0.52, -0.49, 1.2])
    advance(4)
    scene["cinder_c700"].data.root_pos_w[:] = torch.tensor(BIN_POSITION) + torch.tensor([0.0, 0.0, 0.10])
    # Losing the installed card prevents completion even with both cards in the bin.
    scene["rtx_5090"].data.root_pos_w[:, 0] += 0.10
    advance(20)
    assert not tracker.is_complete().any()
    scene["rtx_5090"].data.root_pos_w[:, 0] -= 0.10
    advance(14)
    assert not tracker.is_complete().any()
    advance(1)
    assert tracker.is_complete().all()
    tracker.reset(torch.tensor([0]))
    assert not tracker.is_complete()[0]
    assert tracker.is_complete()[1]
    return True


def test_pc_upgrade_order_and_reset():
    assert run_function_with_persistent_simulation_app(_test_pc_upgrade_order_and_reset)
