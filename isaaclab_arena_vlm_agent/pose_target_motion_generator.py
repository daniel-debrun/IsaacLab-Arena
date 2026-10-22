# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hold absolute pose targets using embodiment-provided state feedback."""

from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena.embodiments.common.command_space import CommandType
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


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
            embodiment: Robot supplying the command layout and measured command state.
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

    def goal_reached(self, goal: np.ndarray, arena_world: ArenaWorld, env_id: int) -> bool:
        """Compare the target to measured translation, rotation, and gripper closure.

        Args:
            goal: Validated target in the embodiment's declared pose layout.
            arena_world: Live Arena scene queries for measured robot state.
            env_id: Environment index within the batched world state.

        Returns:
            Whether every pose and binary control is within its configured tolerance.
        """
        space = self.embodiment.command_space
        assert space is not None and space.command_type == self.command_type
        assert space.quaternion_offsets, "Declare at least one pose quaternion"
        measured = self.embodiment.get_command_state(arena_world)[env_id].detach().cpu().numpy()
        assert measured.shape == goal.shape == (space.dimension,)
        if not np.isfinite(measured).all():
            return False
        checked_indices = set(space.binary_values)
        for offset in space.quaternion_offsets:
            pose_start = offset - 3
            assert 0 <= pose_start and offset + 4 <= space.dimension
            checked_indices.update(range(pose_start, offset + 4))
            # Euclidean distance in meters; both positions are expressed in the robot-root frame R.
            position_error = np.linalg.norm(goal[pose_start:offset] - measured[pose_start:offset])
            # E is the measured end-effector frame; G is its goal frame, both relative to R.
            q_R_E = measured[offset : offset + 4]
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
        assert checked_indices == set(range(space.dimension)), "Expected only pose and binary targets"
        for index, (open_value, closed_value) in space.binary_values.items():
            # Normalize by the native input range; retain measured physical overshoot.
            gripper_error = abs(float(goal[index]) - measured[index]) / abs(closed_value - open_value)
            if gripper_error > self.gripper_tolerance:
                return False
        return True

    def compute_command(self, goal: np.ndarray, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Hold the absolute target while the environment's IK controller tracks measured state.

        Args:
            goal: Validated target in the embodiment's declared pose layout.
            arena_world: Live Arena scene queries for measured robot state.
            env_id: Environment index within the batched world state.

        Returns:
            An independent absolute pose command for the next environment step.
        """
        return goal.copy()
