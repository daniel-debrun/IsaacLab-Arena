# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hold absolute pose targets using embodiment-provided state feedback."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena.embodiments.common.command_type import CommandType
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


class PoseTargetMotionGenerator(MotionGenerator):
    """Hold an absolute IK target until measured pose and gripper tolerances are met."""

    command_type = CommandType.EE_POSE

    def __init__(
        self,
        embodiment: EmbodimentBase,
        position_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.1,
        gripper_tolerance: float = 0.1,
    ):
        """Configure pose tracking and normalized binary-control completion tolerances.

        Args:
            embodiment: Robot supplying root-relative command-body poses, gripper measurements,
                and command metadata. Multiple poses follow quaternion-index order.
            position_tolerance_m: Maximum translation error per pose.
            rotation_tolerance_rad: Maximum shortest rotation error per pose.
            gripper_tolerance: Maximum error as a fraction of each binary control's range.
        """
        assert math.isfinite(position_tolerance_m) and position_tolerance_m > 0
        assert math.isfinite(rotation_tolerance_rad) and 0 < rotation_tolerance_rad <= math.pi
        assert math.isfinite(gripper_tolerance) and 0 < gripper_tolerance < 0.5
        self.embodiment = embodiment
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.gripper_tolerance = gripper_tolerance

    def goal_reached(self, goal: np.ndarray, env, env_id: int) -> bool:
        """Compare the target to measured translation, rotation, and gripper closure.

        Args:
            goal: Validated target in native controller order.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            Whether every pose and binary control is within its configured tolerance.
        """
        assert self.embodiment.get_command_type() == self.command_type, "Expected absolute pose controllers"
        manager = env.unwrapped.action_manager
        quaternion_offsets = self.embodiment.get_quaternion_indices()
        binary_values = self.embodiment.get_binary_action_values(env)
        poses = (
            self.embodiment.get_measured_command_body_pose_in_root(env)[env_id].detach().cpu().numpy().reshape(-1, 7)
        )
        assert goal.shape == (manager.total_action_dim,)
        assert poses.shape == (len(quaternion_offsets), 7)
        if not np.isfinite(poses).all():
            return False
        for pose, offset in zip(poses, quaternion_offsets):
            pose_start = offset - 3
            # Euclidean distance in meters; both positions are expressed in the robot-root frame R.
            position_error = np.linalg.norm(goal[pose_start:offset] - pose[:3])
            # E is the measured end-effector frame; G is its goal frame, both relative to R.
            q_R_E = pose[3:7]
            q_R_G = goal[offset : offset + 4]
            norm_product = np.linalg.norm(q_R_E) * np.linalg.norm(q_R_G)
            if norm_product == 0:
                return False
            # The normalized dot product gives cos(theta / 2); abs handles equivalent q and -q rotations.
            cosine_half_angle = abs(np.dot(q_R_E, q_R_G) / norm_product)
            # Recover the shortest angle in radians; clipping protects arccos from roundoff.
            rotation_error = 2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0))
            if position_error > self.position_tolerance_m or rotation_error > self.rotation_tolerance_rad:
                return False
        action_names = {}
        action_index = 0
        for name, dimension in zip(manager.active_terms, manager.action_term_dim):
            action_names[action_index] = name
            action_index += dimension
        for index, (open_value, closed_value) in binary_values.items():
            closure = self._get_gripper_closure_fraction(env, action_names[index], env_id)
            if not math.isfinite(closure):
                return False
            # Normalize the target to 0=open and 1=closed; retain measured physical overshoot.
            target_closure = (float(goal[index]) - open_value) / (closed_value - open_value)
            gripper_error = abs(target_closure - closure)
            if gripper_error > self.gripper_tolerance:
                return False
        return True

    def _get_gripper_closure_fraction(self, env, action_name: str, env_id: int) -> float:
        """Return dimensionless joint-travel closure: 0=open, 1=closed, with overshoot preserved."""
        gripper = self.embodiment.get_gripper()
        measured = gripper.get_measured_joint_position(env, action_name)[env_id].item()
        open_position, closed_position = gripper.get_open_close_joint_positions(env, action_name)
        return (measured - open_position) / (closed_position - open_position)

    def compute_command(self, goal: np.ndarray, env, env_id: int) -> np.ndarray:
        """Hold the absolute target while the environment's IK controller tracks measured state.

        Args:
            goal: Validated target in native controller order.
            env: Arena environment, optionally gym-wrapped.
            env_id: Environment index within the batched world state.

        Returns:
            An independent absolute pose command for the next environment step.
        """
        return goal.copy()
