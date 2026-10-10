# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered disposal of loose grounds followed by restoration of the machine and bin."""

import torch
from functools import partial

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.composite_task_base import SubtaskSuccessRateMetric
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.coffee_disposal import layout


def joint_state(env, asset_name, joint_name):
    """Return a named articulation joint's position and velocity for every environment."""
    asset = env.scene[asset_name]
    index = asset.find_joints(joint_name)[0][0]
    return asset.data.joint_pos[:, index], asset.data.joint_vel[:, index]


def joint_at(env, asset_name, joint_name, target, tolerance=0.012):
    """Require a named slider to reach its target and stop."""
    position, velocity = joint_state(env, asset_name, joint_name)
    return ((position - target).abs() < tolerance) & (velocity.abs() < 0.035)


def spout_raised(env):
    """Keep the dispenser raised while removing and restoring the tray."""
    return joint_at(env, "coffee_machine", "spout_slide", layout.SPOUT_RAISED, 0.010)


def tray_extracted(env):
    """Require the tray to be fully extended with the dispenser clear."""
    return spout_raised(env) & joint_at(env, "coffee_machine", "tray_slide", layout.TRAY_EXTENSION)


def drawer_open(env):
    """Require the waste drawer to be fully extended and stationary."""
    return joint_at(env, "waste_cabinet", "drawer_slide", layout.DRAWER_EXTENSION)


def support_position(env, asset_name, joint_name, offset):
    """Return a support's world position, including its parent slider displacement."""
    asset = env.scene[asset_name].data
    displacement, _ = joint_state(env, asset_name, joint_name)
    t_A_S = asset.root_pos_w[:].new_tensor(offset).expand(env.num_envs, 3).clone()
    t_A_S[:, 1] += displacement
    return asset.root_pos_w[:] + quat_apply(asset.root_quat_w[:], t_A_S)


def cup_seat_position(env):
    """Return the grounds-container seat on the moving drip tray."""
    offset = tuple(a + b for a, b in zip(layout.TRAY_IN_MACHINE, layout.CUP_IN_TRAY))
    return support_position(env, "coffee_machine", "tray_slide", offset)


def bin_position(env):
    """Return the organic bin's bottom frame origin on the waste drawer."""
    return support_position(env, "waste_cabinet", "drawer_slide", layout.BIN_IN_DRAWER)


def lid_position(env, parked=False):
    """Return the lid's home or temporary-support position on the moving drawer."""
    offset = layout.LID_PARK_IN_DRAWER if parked else layout.LID_IN_DRAWER
    return support_position(env, "waste_cabinet", "drawer_slide", offset)


def settled(velocity):
    """Require low linear and angular speed before accepting a supported object."""
    return (torch.linalg.vector_norm(velocity[:, :3], dim=-1) < 0.06) & (
        torch.linalg.vector_norm(velocity[:, 3:], dim=-1) < 0.35
    )


def released(env, name):
    """Require release of this object and clearance from the observed robot hand."""
    clear = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    grasps = getattr(env, "coffee_grasps", {})
    if name in grasps:
        clear &= ~grasps[name]
    if "ee_frame" in env.scene.keys():
        hand = env.scene["ee_frame"].data.target_pos_w[:, 0]
        clear &= torch.linalg.vector_norm(hand - env.scene[name].data.root_pos_w[:], dim=-1) > 0.14
    return clear


def supported(env, name, target, support_name):
    """Require aligned, settled, released placement within the support's seat."""
    obj = env.scene[name].data
    support = env.scene[support_name].data
    t_S_O = quat_apply_inverse(support.root_quat_w[:], obj.root_pos_w[:] - target)
    aligned = (obj.root_quat_w[:] * support.root_quat_w[:]).sum(dim=-1).abs() > 0.9962
    in_seat = (t_S_O[:, :2].abs() < 0.014).all(dim=-1) & (t_S_O[:, 2].abs() < 0.012)
    return in_seat & aligned & settled(obj.root_vel_w[:]) & released(env, name)


def cup_seated(env):
    """Require the upright grounds container to rest in the drip-tray recess."""
    return supported(env, "grounds_container", cup_seat_position(env), "coffee_machine")


def tray_extracted_with_cup(env):
    """Observe the complete supported tray assembly after extraction."""
    return tray_extracted(env) & cup_seated(env)


def lid_home(env):
    """Require the lid to be seated and released on the organic bin."""
    return supported(env, "bin_lid", lid_position(env), "waste_cabinet")


def lid_parked(env):
    """Require the removed lid to rest on the neighboring bin support."""
    return drawer_open(env) & supported(env, "bin_lid", lid_position(env, parked=True), "waste_cabinet")


def lid_lifted(env, parked=False):
    """Observe a lid lifted clear of its current support before accepting placement."""
    obj = env.scene["bin_lid"].data
    cabinet = env.scene["waste_cabinet"].data
    t_D_L = quat_apply_inverse(cabinet.root_quat_w[:], obj.root_pos_w[:] - lid_position(env, parked))
    return drawer_open(env) & (t_D_L[:, :2].abs() < 0.08).all(dim=-1) & (t_D_L[:, 2] > 0.065)


def cup_lifted(env):
    """Observe the cup lifted out of the extracted tray after the bin is uncovered."""
    cup = env.scene["grounds_container"].data
    machine = env.scene["coffee_machine"].data
    t_T_C = quat_apply_inverse(machine.root_quat_w[:], cup.root_pos_w[:] - cup_seat_position(env))
    return tray_extracted(env) & lid_parked(env) & (t_T_C[:, 2] > 0.12)


def cup_above_bin(env):
    """Require the cup above the open organic-bin aperture, before and during tipping."""
    cup = env.scene["grounds_container"].data
    cabinet = env.scene["waste_cabinet"].data
    t_B_C = quat_apply_inverse(cabinet.root_quat_w[:], cup.root_pos_w[:] - bin_position(env))
    over_opening = (t_B_C[:, 0].abs() < 0.075) & (t_B_C[:, 1].abs() < 0.11)
    above_rim = (t_B_C[:, 2] > layout.BIN_HEIGHT + 0.035) & (t_B_C[:, 2] < layout.BIN_HEIGHT + 0.45)
    return drawer_open(env) & lid_parked(env) & over_opening & above_rim


def grounds_in_cup(env):
    """Observe all loose clumps inside the upright cup before its tipping stage."""
    cup = env.scene["grounds_container"].data
    inside = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in layout.GROUND_NAMES:
        t_C_G = quat_apply_inverse(cup.root_quat_w[:], env.scene[name].data.root_pos_w[:] - cup.root_pos_w[:])
        inside &= (t_C_G[:, 0].abs() < layout.CUP_SIZE[0] / 2) & (t_C_G[:, 1].abs() < layout.CUP_SIZE[1] / 2)
        inside &= (t_C_G[:, 2] > 0.0) & (t_C_G[:, 2] < layout.CUP_SIZE[2] + layout.GROUND_RADIUS)
    return inside


def tipped_over_bin(env):
    """Observe a downward-facing cup mouth (over 110 degrees) above the organic bin."""
    quaternion = env.scene["grounds_container"].data.root_quat_w[:]
    up_z = 1.0 - 2.0 * (quaternion[:, 0].square() + quaternion[:, 1].square())
    return cup_above_bin(env) & (up_z < -0.3421)


def grounds_contained(env):
    """Require every vertex of each clump's collision hull inside the moving bin."""
    cabinet = env.scene["waste_cabinet"].data
    origin = bin_position(env)
    vertices_G = origin.new_tensor(layout.GROUND_COLLISION_VERTICES).expand(env.num_envs, -1, -1)
    half_width = origin.new_tensor(layout.BIN_HALF_INTERIOR)
    # The explicit collision mesh avoids PhysX's detailed-mesh hull simplification.
    # Only a 0.2 mm allowance remains for solver contact and floating-point error.
    tolerance = 0.0002
    contained = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in layout.GROUND_NAMES:
        ground = env.scene[name].data
        # B and G denote the bin and clump frames; the bin follows the drawer.
        t_B_G, q_B_G = subtract_frame_transforms(
            origin, cabinet.root_quat_w[:], ground.root_pos_w[:], ground.root_quat_w[:]
        )
        rotations = q_B_G[:, None, :].expand(-1, vertices_G.shape[1], -1)
        vertices_B = quat_apply(rotations, vertices_G) + t_B_G[:, None, :]
        contained &= (vertices_B[:, :, :2].abs() <= half_width + tolerance).all(dim=(1, 2))
        contained &= (vertices_B[:, :, 2] >= layout.BIN_FLOOR_THICKNESS - tolerance).all(dim=1)
        contained &= (vertices_B[:, :, 2] <= layout.BIN_HEIGHT + tolerance).all(dim=1)
    return contained


def discharge_started(env):
    """Observe meaningful tipping while at least one clump has yet to enter the bin."""
    quaternion = env.scene["grounds_container"].data.root_quat_w[:]
    up_z = 1.0 - 2.0 * (quaternion[:, 0].square() + quaternion[:, 1].square())
    # Grounds can leave this shallow cup before the eventual 110-degree inversion.
    tilted = up_z < 0.819152  # cos(35 degrees)
    return cup_above_bin(env) & tilted & ~grounds_contained(env)


def grounds_in_bin(env):
    """Require all eight collision hulls inside the bin and settled below its rim."""
    contained = grounds_contained(env)
    vertices_G = contained.new_tensor(layout.GROUND_COLLISION_VERTICES, dtype=torch.float32)
    vertices_G = vertices_G.expand(env.num_envs, -1, -1)
    for name in layout.GROUND_NAMES:
        ground = env.scene[name].data
        velocity = ground.root_vel_w[:]
        rotations = ground.root_quat_w[:][:, None, :].expand(-1, vertices_G.shape[1], -1)
        radii_W = quat_apply(rotations, vertices_G)
        angular_velocity = velocity[:, None, 3:].expand_as(radii_W)
        vertex_velocity = velocity[:, None, :3] + torch.linalg.cross(angular_velocity, radii_W, dim=-1)
        # Use the actual clump size: cup/lid angular limits misclassify tiny contact
        # correction velocities. Center drift is held to a stricter 1 cm/s limit.
        contained &= torch.linalg.vector_norm(velocity[:, :3], dim=-1) < 0.01
        contained &= (torch.linalg.vector_norm(vertex_velocity, dim=-1) < 0.02).all(dim=1)
    return contained


def disposal_complete(env):
    """Keep the bin accessible until all loose grounds have landed and settled."""
    return drawer_open(env) & lid_parked(env) & grounds_in_bin(env)


def cup_returned(env):
    """Seat the empty cup while its tray is still extracted and the bin remains open."""
    return tray_extracted(env) & cup_seated(env) & disposal_complete(env)


def tray_inserted(env):
    """Restore the complete tray assembly while keeping the grounds in the open bin."""
    return (
        spout_raised(env) & joint_at(env, "coffee_machine", "tray_slide", 0.0) & cup_seated(env) & grounds_in_bin(env)
    )


def lid_replaced(env):
    """Restore the bin lid after reinserting the machine tray, before closing the drawer."""
    return drawer_open(env) & lid_home(env) & tray_inserted(env)


def final_state(env):
    """Recheck every retained result with the drawer closed and both objects released."""
    return joint_at(env, "waste_cabinet", "drawer_slide", 0.0) & lid_home(env) & tray_inserted(env)


def dropped_object(env):
    """Fail on a cup, lid, or loose clump reaching the floor; allow normal bin descent."""
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ("grounds_container", "bin_lid", *layout.GROUND_NAMES):
        height = env.scene[name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        dropped |= height < 0.065
    return dropped


def reset_joints(env, env_ids):
    """Reset the selected environments' three sliders and their drive targets."""
    for name in ("coffee_machine", "waste_cabinet"):
        asset = env.scene[name]
        position = torch.zeros_like(asset.data.joint_pos[env_ids])
        asset.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=env_ids)
        asset.set_joint_position_target(position, env_ids=env_ids)


def object_state(env):
    """Expose environment-local object poses, velocities, and named mechanism states."""
    states = []
    for name in ("coffee_machine", "waste_cabinet", "grounds_container", "bin_lid", *layout.GROUND_NAMES):
        data = env.scene[name].data
        states.extend((data.root_pos_w[:] - env.scene.env_origins, data.root_quat_w[:], data.root_vel_w[:]))
    for name in ("coffee_machine", "waste_cabinet"):
        states.extend((env.scene[name].data.joint_pos[:], env.scene[name].data.joint_vel[:]))
    return torch.cat(states, dim=-1)


@configclass
class CoffeeDisposalEventsCfg:
    reset_coffee_sliders = EventTermCfg(func=reset_joints, mode="reset")


@configclass
class CoffeeDisposalObservationsCfg:
    @configclass
    class TaskCfg(ObservationGroupCfg):
        objects = ObservationTermCfg(func=object_state)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    task: TaskCfg = TaskCfg()


class CoffeeDisposalTask(TaskBase):
    """Dispose of loose spent grounds and restore the machine, bin lid, and waste drawer."""

    def __init__(self, episode_length_s=300.0):
        super().__init__(
            episode_length_s,
            "Raise the coffee spout, extract the drip tray, open the waste drawer, park the organic-bin lid, lift and"
            " empty the grounds container into the organic bin, return it to the tray, insert the tray, replace the"
            " lid, and close the waste drawer.",
        )

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return CoffeeDisposalEventsCfg()

    def get_observation_cfg(self):
        return CoffeeDisposalObservationsCfg()

    def get_termination_cfg(self):
        dwell = TrueForConsecutiveStepsCfg
        home_tray = partial(joint_at, asset_name="coffee_machine", joint_name="tray_slide", target=0.0)
        closed_drawer = partial(joint_at, asset_name="waste_cabinet", joint_name="drawer_slide", target=0.0)
        low_spout = partial(joint_at, asset_name="coffee_machine", joint_name="spout_slide", target=0.0)
        stages = [
            ("raise_spout", [low_spout, dwell(spout_raised, 3)]),
            ("extract_tray", [home_tray, dwell(tray_extracted_with_cup, 5)]),
            ("open_waste_drawer", [closed_drawer, dwell(drawer_open, 5)]),
            ("park_bin_lid", [lid_home, dwell(lid_lifted, 3), dwell(lid_parked, 8)]),
            ("lift_grounds_container", [cup_seated, dwell(cup_lifted, 3)]),
            ("carry_above_bin", [dwell(cup_above_bin, 3)]),
            (
                "dispose_grounds",
                [grounds_in_cup, discharge_started, dwell(tipped_over_bin, 3), dwell(disposal_complete, 8)],
            ),
            ("return_grounds_container", [dwell(cup_returned, 8)]),
            ("insert_tray", [tray_extracted, dwell(tray_inserted, 8)]),
            ("replace_bin_lid", [lid_parked, dwell(partial(lid_lifted, parked=True), 3), dwell(lid_replaced, 8)]),
            ("close_waste_drawer", [drawer_open, dwell(final_state, 15)]),
        ]
        success = []
        for index, (name, sequence) in enumerate(stages):
            success.append(CompletionCriteria(name=name, parent_subtask_idx=index, predicate_sequence=sequence))
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=success,
            failures={"dropped_coffee_object": TerminationTermCfg(func=dropped_object)},
            subtasks_are_sequential=True,
        )

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric(), SubtaskSuccessRateMetric()]

    def get_viewer_cfg(self):
        return ViewerCfg(eye=(2.3, -3.2, 2.2), lookat=(0.1, -0.15, 0.85), resolution=(1440, 1080))
