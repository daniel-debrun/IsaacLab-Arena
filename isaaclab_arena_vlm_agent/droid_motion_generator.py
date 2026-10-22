# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID absolute-pose motion generation using live robot state."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.commands import CommandType
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


class DroidPoseMotionGenerator(MotionGenerator):
    """Hold an absolute IK target until measured pose and gripper tolerances are met."""

    command_type = CommandType.EE_POSE

    def __init__(
        self,
        position_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.1,
        gripper_tolerance: float = 0.1,
        gripper_closed_position_rad: float = math.pi / 4,
    ):
        assert math.isfinite(position_tolerance_m) and position_tolerance_m > 0
        assert math.isfinite(rotation_tolerance_rad) and 0 < rotation_tolerance_rad <= math.pi
        assert math.isfinite(gripper_tolerance) and 0 < gripper_tolerance < 0.5
        assert math.isfinite(gripper_closed_position_rad) and gripper_closed_position_rad > 0
        self.gripper_closed_position_rad = gripper_closed_position_rad
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.gripper_tolerance = gripper_tolerance

    def goal_reached(self, goal: np.ndarray, arena_world: ArenaWorld, env_id: int) -> bool:
        """Compare the target to measured translation, rotation, and gripper closure.

        Args:
            goal: Validated eight-element absolute target.
            arena_world: Live Arena scene queries for measured pose and finger position.
            env_id: Environment index within the batched world state.

        Returns:
            Whether all three errors are within their configured tolerances.
        """
        T_R_E = arena_world.get_body_pose_in_root("robot", "base_link")[env_id].detach().cpu().numpy()
        finger_position = arena_world.get_joint_position("robot", "finger_joint")[env_id].item()
        gripper_closed_fraction = finger_position / self.gripper_closed_position_rad
        # Euclidean distance in meters; both positions are expressed in the robot-root frame R.
        position_error = np.linalg.norm(goal[:3] - T_R_E[:3])
        # E is the measured end-effector frame; G is its goal frame, both relative to R.
        q_R_E = T_R_E[3:]
        q_R_G = goal[3:7]
        # The normalized dot product gives cos(theta / 2); abs handles equivalent q and -q rotations.
        cosine_half_angle = abs(np.dot(q_R_E, q_R_G) / (np.linalg.norm(q_R_E) * np.linalg.norm(q_R_G)))
        # Recover the shortest rotation angle in radians; clipping prevents roundoff from invalidating arccos.
        rotation_error = 2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0))
        # Compare the binary target (0 open, 1 closed) with measured closure, including physical overshoot.
        gripper_error = abs(float(goal[7]) - gripper_closed_fraction)
        return bool(
            position_error <= self.position_tolerance_m
            and rotation_error <= self.rotation_tolerance_rad
            and gripper_error <= self.gripper_tolerance
        )

    def compute_command(self, goal: np.ndarray, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Hold the absolute target while the environment's IK controller tracks measured state.

        Args:
            goal: Validated eight-element absolute target.
            arena_world: Live Arena scene queries for measured pose and finger position.
            env_id: Environment index within the batched world state.

        Returns:
            An independent absolute pose command for the next environment step.
        """
        return goal.copy()
