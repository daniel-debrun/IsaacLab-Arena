# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Track absolute pose targets and grippers using live controller and scene state."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.command_indices import read_absolute_command_indices
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gripper import Gripper


class PoseTargetMotionGenerator(MotionGenerator):
    """Hold absolute pose targets until pose and gripper completion conditions are met."""

    def __init__(
        self,
        grippers: dict[str, Gripper] | None = None,
        position_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.1,
        gripper_tolerance_m: float = 0.001,
        stall_margin_m: float = 0.001,
        stall_motion_tolerance_m: float = 0.0001,
        stall_steps: int = 3,
    ):
        """Configure pose accuracy and closed-gripper stall detection.

        Args:
            grippers: Gripper measurements keyed by binary action term name; omit for pose-only robots.
            position_tolerance_m: Maximum translation error per pose.
            rotation_tolerance_rad: Maximum shortest rotation error per pose.
            gripper_tolerance_m: Maximum absolute measured-minus-commanded finger displacement.
            stall_margin_m: Minimum closing error for a gripper blocked before its closed target.
            stall_motion_tolerance_m: Maximum change in opening width per step during a stall.
            stall_steps: Consecutive stationary steps required to accept a blocked closing goal.
        """
        assert math.isfinite(position_tolerance_m) and position_tolerance_m > 0
        assert math.isfinite(rotation_tolerance_rad) and 0 < rotation_tolerance_rad <= math.pi
        assert math.isfinite(gripper_tolerance_m) and gripper_tolerance_m > 0
        assert math.isfinite(stall_margin_m) and stall_margin_m >= gripper_tolerance_m
        assert math.isfinite(stall_motion_tolerance_m) and stall_motion_tolerance_m >= 0
        assert type(stall_steps) is int and stall_steps > 0
        self.grippers = dict(grippers or {})
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.gripper_tolerance_m = gripper_tolerance_m
        self.stall_margin_m = stall_margin_m
        self.stall_motion_tolerance_m = stall_motion_tolerance_m
        self.stall_steps = stall_steps
        self._gripper_history: dict[tuple[int, int, str], tuple[bytes, int, float, int]] = {}

    def goal_reached(self, goal: np.ndarray, env, env_id: int) -> bool:
        """Check measured poses and accept grippers at target or stably blocked while closing.

        Args:
            goal: Validated absolute target in native controller order.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            Whether all poses and grippers have completed the requested command.
        """
        indices = self._command_indices(goal, env)
        poses_reached = self._poses_reached(goal, env, env_id, indices)
        grippers_reached = self._grippers_reached(goal, env, env_id, indices)
        return poses_reached and grippers_reached

    def _command_indices(self, goal, env):
        """Check that the target uses supported pose controllers and known gripper measurements."""
        indices = read_absolute_command_indices(env)
        assert indices is not None and indices.poses and not indices.joints, "Expected absolute pose controllers"
        assert goal.shape == (indices.action_dim,)
        for command in indices.grippers:
            assert command.term_name in self.grippers, f"Provide gripper measurements for {command.term_name}"
        return indices

    def _poses_reached(self, goal, env, env_id, indices) -> bool:
        """Compare controlled bodies to their root-relative XYZ/XYZW targets."""
        for command in indices.poses:
            pose = env.unwrapped.arena_world.get_body_pose_in_root(command.asset_name, command.body_name)
            pose = pose[env_id].detach().cpu().numpy()
            target = goal[command.columns]
            if not np.isfinite(pose).all():
                return False
            if _position_error_m(pose[:3], target[:3]) > self.position_tolerance_m:
                return False
            if _rotation_error_rad(pose[3:], target[3:]) > self.rotation_tolerance_rad:
                return False
        return True

    def _grippers_reached(self, goal, env, env_id, indices) -> bool:
        """Use existing gripper measurements, requiring sustained stillness for blocked closing."""
        base = env.unwrapped
        reached = True
        for command in indices.grippers:
            gripper = self.grippers[command.term_name]
            error = float(gripper.get_closing_error_m(base)[env_id])
            gap = float(gripper.get_opening_width_m(base.arena_world)[env_id])
            step = int(base.episode_length_buf[env_id])
            key = (id(base), env_id, command.term_name)
            previous = self._gripper_history.get(key)
            target = goal.tobytes()
            stationary_steps = 0
            if previous is not None:
                previous_target, previous_step, previous_gap, previous_count = previous
                if target == previous_target and step == previous_step:
                    stationary_steps = previous_count
                elif target == previous_target and step == previous_step + 1:
                    if abs(gap - previous_gap) <= self.stall_motion_tolerance_m:
                        stationary_steps = previous_count + 1
            self._gripper_history[key] = (target, step, gap, stationary_steps)
            # Closing error refers to the last applied command, which must match this goal.
            applied = float(base.action_manager.action[env_id, command.column])
            if not (math.isfinite(error) and math.isfinite(gap) and applied == goal[command.column]):
                reached = False
                continue
            at_target = abs(error) <= self.gripper_tolerance_m
            blocked_closing = (
                goal[command.column] == command.close_value
                and error > self.stall_margin_m
                and stationary_steps >= self.stall_steps
            )
            reached = reached and (at_target or blocked_closing)
        return reached

    def compute_command(self, goal: np.ndarray, env, env_id: int) -> np.ndarray:
        """Return an independent copy of the absolute pose command to hold.

        Args:
            goal: Validated target in native controller order.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            The absolute target for the next environment step.
        """
        self._command_indices(goal, env)
        return goal.copy()


def _position_error_m(measured: np.ndarray, target: np.ndarray) -> float:
    """Return Euclidean translation error in meters."""
    # Both positions are expressed in the robot-root frame R.
    return float(np.linalg.norm(target - measured))


def _rotation_error_rad(q_R_E: np.ndarray, q_R_G: np.ndarray) -> float:
    """Return the shortest rotation error in radians, or infinity for a zero quaternion."""
    # E is the measured end-effector frame; G is its goal frame, both relative to root R.
    norm_product = np.linalg.norm(q_R_E) * np.linalg.norm(q_R_G)
    if norm_product == 0:
        return math.inf
    # The normalized dot product gives cos(theta / 2); abs handles equivalent q and -q rotations.
    cosine_half_angle = abs(np.dot(q_R_E, q_R_G) / norm_product)
    # Recover the shortest angle in radians; clipping protects arccos from roundoff.
    return float(2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0)))
