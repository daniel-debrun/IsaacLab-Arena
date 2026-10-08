# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered, reset-safe PC service task using Arena's progress tracker."""

import torch
from functools import partial

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.composite_task_base import SubtaskSuccessRateMetric
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.pc_upgrade.geometry import contained_card, hand_clear_of_card, seated_card, settled
from isaaclab_arena_environments.pc_upgrade.layout import BIN_POSITION, SLOT_IN_PC, SLOT_QUATERNION, TABLE_Z


def panel_open(env):
    """Return whether the panel has opened at least 80 degrees."""
    return env.scene["pc_chassis"].data.joint_pos[:, 0] < -1.3963


def panel_closed(env):
    """Return whether the panel is within three degrees of closed and stopped."""
    data = env.scene["pc_chassis"].data
    return (data.joint_pos[:, 0].abs() < 0.0524) & (data.joint_vel[:, 0].abs() < 0.1)


def at_bench(env, name):
    """Observe a settled card at bench height before crediting its next lift."""
    data = env.scene[name].data
    position = data.root_pos_w - env.scene.env_origins
    relative = position - position.new_tensor(BIN_POSITION)
    outside_bin = (relative[:, 0].abs() > 0.23) | (relative[:, 1].abs() > 0.19)
    height = position[:, 2] - TABLE_Z
    return outside_bin & (height > 0.0) & (height < 0.06) & settled(data.root_vel_w)


def lifted(env):
    """Require the RTX card to clear the bench with the panel open."""
    height = env.scene["rtx_5090"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    position = env.scene["rtx_5090"].data.root_pos_w - env.scene["pc_chassis"].data.root_pos_w
    outside_pc = (position[:, 0].abs() > 0.23) | (position[:, 1].abs() > 0.14)
    return panel_open(env) & outside_pc & (height > TABLE_Z + 0.10)


def socket_pose(env):
    """Return the card pose relative to the motherboard socket, in metres and xyzw."""
    pc = env.scene["pc_chassis"].data
    gpu = env.scene["rtx_5090"].data
    # P, S and G denote the PC, socket and graphics-card frames.
    t_P_G, q_P_G = subtract_frame_transforms(pc.root_pos_w[:], pc.root_quat_w[:], gpu.root_pos_w[:], gpu.root_quat_w[:])
    t_P_S = t_P_G.new_tensor(SLOT_IN_PC).expand_as(t_P_G)
    q_P_S = q_P_G.new_tensor(SLOT_QUATERNION).expand_as(q_P_G)
    return subtract_frame_transforms(t_P_S, q_P_S, t_P_G, q_P_G)


def insertion_approach(env):
    """Observe an aligned card outside the side opening before entering the socket."""
    t_S_G, q_S_G = socket_pose(env)
    # Socket frame S rotates 180 degrees about X: its +Y points out of the case.
    aligned = q_S_G[:, 3].abs() > 0.9962
    in_front = (t_S_G[:, 1] > 0.12) & (t_S_G[:, 1] < 0.25)
    on_axis = (t_S_G[:, 0].abs() < 0.012) & (t_S_G[:, 2].abs() < 0.012)
    return panel_open(env) & aligned & in_front & on_axis


def installed(env):
    """Require the correct card to be aligned, stationary, and clear of the hand."""
    gpu = env.scene["rtx_5090"].data
    t_S_G, q_S_G = socket_pose(env)
    hand = env.scene["ee_frame"].data.target_pos_w[:, 0]
    t_G_H = quat_apply_inverse(gpu.root_quat_w[:], hand - gpu.root_pos_w)
    hand_clear = hand_clear_of_card(t_G_H)
    return seated_card(t_S_G, q_S_G, gpu.root_vel_w) & hand_clear


def ready_to_recycle(env, name):
    """Require a fresh lift outside the bin after the preceding stage completes."""
    position = env.scene[name].data.root_pos_w - env.scene.env_origins
    relative = position - position.new_tensor(BIN_POSITION)
    outside_bin = (relative[:, 0].abs() > 0.23) | (relative[:, 1].abs() > 0.19)
    return outside_bin & (position[:, 2] > TABLE_Z + 0.12) & panel_closed(env) & installed(env)


def insertion_complete(env):
    """Seat and release the GPU while the chassis is still open."""
    return installed(env) & panel_open(env)


def closed_after_install(env):
    """Close the panel while keeping the installed card seated."""
    return panel_closed(env) & installed(env)


def recycled(env, name):
    """Require an entire fictional card to rest below the recycling bin rim."""
    gpu = env.scene[name].data
    t_B_G = gpu.root_pos_w - env.scene.env_origins - gpu.root_pos_w.new_tensor(BIN_POSITION)
    return contained_card(t_B_G, gpu.root_quat_w, gpu.root_vel_w)


def final_state(env):
    """Recheck every persistent result when completing the final stage."""
    return installed(env) & panel_closed(env) & recycled(env, "vortex_vx880") & recycled(env, "cinder_c700")


def dropped_card(env):
    """End an episode when any GPU falls below the tabletop."""
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ("rtx_5090", "vortex_vx880", "cinder_c700"):
        dropped |= env.scene[name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < TABLE_Z - 0.15
    return dropped


def reset_panel(env, env_ids):
    """Restore the selected chassis hinges to rest without changing other environments."""
    pc = env.scene["pc_chassis"]
    position = torch.zeros_like(pc.data.joint_pos[env_ids])
    pc.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=env_ids)
    pc.set_joint_position_target(position, env_ids=env_ids)


def object_state(env):
    """Expose local poses and hinge state for state-based controllers."""
    states = []
    for name in ("rtx_5090", "vortex_vx880", "cinder_c700", "pc_chassis"):
        data = env.scene[name].data
        states.extend((data.root_pos_w - env.scene.env_origins, data.root_quat_w))
    states.append(env.scene["pc_chassis"].data.joint_pos)
    return torch.cat(states, dim=-1)


@configclass
class PCUpgradeEventsCfg:
    reset_service_panel = EventTermCfg(func=reset_panel, mode="reset")


@configclass
class PCUpgradeObservationsCfg:
    @configclass
    class TaskCfg(ObservationGroupCfg):
        objects = ObservationTermCfg(func=object_state)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    task: TaskCfg = TaskCfg()


class PCUpgradeTask(TaskBase):
    """Open, select, install, close, and recycle with one shared five-minute budget."""

    def __init__(self, episode_length_s=300.0):
        super().__init__(
            episode_length_s,
            "Open the PC chassis, pick up the NVIDIA GeForce RTX 5090, insert it into the motherboard PCIe slot, close"
            " the chassis, then put the VORTEX VX 880 and CINDER C700 cards in the electronics recycling bin.",
        )

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return PCUpgradeEventsCfg()

    def get_observation_cfg(self):
        return PCUpgradeObservationsCfg()

    def get_termination_cfg(self):
        stages = [
            ("open_chassis", panel_open, 5),
            ("pick_rtx_5090", lifted, 3),
            ("install_rtx_5090", insertion_complete, 10),
            ("close_chassis", closed_after_install, 10),
            ("recycle_vortex", partial(recycled, name="vortex_vx880"), 10),
            ("recycle_cinder", final_state, 15),
        ]
        success = []
        for index, (name, predicate, steps) in enumerate(stages):
            sequence = [TrueForConsecutiveStepsCfg(predicate, steps)]
            if index == 1:
                sequence.insert(0, partial(at_bench, name="rtx_5090"))
            if index == 2:
                sequence.insert(0, TrueForConsecutiveStepsCfg(insertion_approach, 3))
            if index >= 4:
                card_name = "vortex_vx880" if index == 4 else "cinder_c700"
                sequence.insert(0, TrueForConsecutiveStepsCfg(partial(ready_to_recycle, name=card_name), 3))
                sequence.insert(0, partial(at_bench, name=card_name))
            success.append(CompletionCriteria(name=name, parent_subtask_idx=index, predicate_sequence=sequence))
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=success,
            failures={"dropped_gpu": TerminationTermCfg(func=dropped_card)},
            subtasks_are_sequential=True,
        )

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric(), SubtaskSuccessRateMetric()]

    def get_viewer_cfg(self):
        return ViewerCfg(eye=(1.65, -1.70, 1.95), lookat=(0.43, 0.03, 0.96), resolution=(1440, 1080))
