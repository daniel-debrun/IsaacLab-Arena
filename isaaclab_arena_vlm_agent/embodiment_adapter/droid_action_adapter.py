# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Absolute DROID pose commands for single-step, chunk, and goal policies."""

import math
import numpy as np

from isaaclab_arena_vlm_agent.embodiment_adapter.action_adapter import (
    AgentActionAdapter,
    ChunkActionAdapter,
    GoalActionAdapter,
)
from isaaclab_arena_vlm_agent.embodiment_adapter.droid_pose import DroidPoseChunk, DroidPoseCommand


def _pose_command(response: DroidPoseCommand) -> np.ndarray:
    """Pack absolute position, XYZW rotation, and binary gripper closure."""
    return np.asarray(
        [*response.position_root_m, *response.quaternion_root_xyzw, float(response.gripper_closed)], dtype=np.float32
    )


class DroidPoseActionAdapter(AgentActionAdapter):
    """Convert absolute base_link targets into unscaled absolute DROID IK actions."""

    response_model = DroidPoseCommand

    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether actions are absolute base_link IK followed by binary DROID gripper control.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

        manager = env.unwrapped.action_manager
        if manager.active_terms != ["arm_action", "gripper_action"]:
            return False
        arm = manager.get_term("arm_action")
        gripper = manager.get_term("gripper_action")
        if not isinstance(arm, DifferentialInverseKinematicsAction) or not isinstance(
            gripper, BinaryJointPositionZeroToOneAction
        ):
            return False
        cfg = arm.cfg
        return bool(
            arm.action_dim == 7
            and gripper.action_dim == 1
            and cfg.asset_name == gripper.cfg.asset_name == "robot"
            and cfg.body_name == "base_link"
            and cfg.controller.command_type == "pose"
            and not cfg.controller.use_relative_mode
            and np.all(np.asarray(cfg.scale) == 1.0)
            and cfg.body_offset is None
            and cfg.clip is None
            and gripper.cfg.clip is None
        )

    def decode_command(self, response: DroidPoseCommand, policy_state: dict) -> np.ndarray:
        """Return the absolute target without applying the measured pose as an offset.

        Args:
            response: Validated target in meters, XYZW, and binary gripper closure.
            policy_state: Measured pose and gripper state accompanying the request.

        Returns:
            Eight-element command: position, quaternion, and zero/open or one/closed.
        """
        return _pose_command(response)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a command has a finite pose, unit quaternion, and binary gripper target.

        Args:
            command: Packed absolute pose and gripper command.
        """
        return bool(
            command.shape == (8,)
            and np.isfinite(command).all()
            and np.isclose(np.linalg.norm(command[3:7]), 1.0, atol=1e-3)
            and command[7] in (0.0, 1.0)
        )


class DroidChunkActionAdapter(ChunkActionAdapter, DroidPoseActionAdapter):
    """Execute a fixed-length sequence of absolute DROID poses, one per environment step."""

    response_model = DroidPoseChunk

    def decode_command(self, response: DroidPoseChunk, policy_state: dict) -> np.ndarray:
        """Pack each validated pose in execution order.

        Args:
            response: Validated sequence of absolute poses and gripper targets.
            policy_state: Measured pose and gripper state accompanying the request.

        Returns:
            Array of shape (number of commands, 8), checked against chunk_size before execution.
        """
        return np.stack([_pose_command(command) for command in response.commands])

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether every row is a valid pose and the chunk has its configured length.

        Args:
            command: Packed command chunk.
        """
        return bool(
            command.shape == (self.chunk_size, 8)
            and all(DroidPoseActionAdapter.is_command_valid(self, row) for row in command)
        )


class DroidGoalActionAdapter(GoalActionAdapter, DroidPoseActionAdapter):
    """Hold an absolute IK target until measured pose and gripper tolerances are met.

    A blocked gripper can exhaust max_steps; issuing a close command alone does not
    establish that the fingers reached the requested closure or that a grasp succeeded.
    """

    def __init__(
        self,
        max_steps: int,
        position_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.1,
        gripper_tolerance: float = 0.1,
    ):
        super().__init__(max_steps)
        assert math.isfinite(position_tolerance_m) and position_tolerance_m > 0
        assert math.isfinite(rotation_tolerance_rad) and 0 < rotation_tolerance_rad <= math.pi
        assert math.isfinite(gripper_tolerance) and 0 < gripper_tolerance < 0.5
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.gripper_tolerance = gripper_tolerance

    def goal_reached(self, goal: np.ndarray, policy_state: dict) -> bool:
        """Compare the target to measured translation, rotation, and gripper closure.

        Args:
            goal: Validated eight-element absolute target.
            policy_state: Current measured pose and gripper closure.

        Returns:
            Whether all three errors are within their configured tolerances.
        """
        # Euclidean distance in meters; both positions are expressed in the robot-root frame R.
        position_error = np.linalg.norm(goal[:3] - np.asarray(policy_state["position_root_m"]))
        # E is the measured end-effector frame; G is its goal frame, both relative to R.
        q_R_E = np.asarray(policy_state["quaternion_root_xyzw"])
        q_R_G = goal[3:7]
        # The normalized dot product gives cos(theta / 2); abs handles equivalent q and -q rotations.
        cosine_half_angle = abs(np.dot(q_R_E, q_R_G) / (np.linalg.norm(q_R_E) * np.linalg.norm(q_R_G)))
        # Recover the shortest rotation angle in radians; clipping prevents roundoff from invalidating arccos.
        rotation_error = 2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0))
        # Compare the binary target (0 open, 1 closed) with measured closure, including physical overshoot.
        gripper_error = abs(float(goal[7]) - policy_state["gripper_closed_fraction"])
        return bool(
            position_error <= self.position_tolerance_m
            and rotation_error <= self.rotation_tolerance_rad
            and gripper_error <= self.gripper_tolerance
        )

    def compute_command(self, goal: np.ndarray, policy_state: dict) -> np.ndarray:
        """Hold the absolute target while the environment's IK controller tracks measured state.

        Args:
            goal: Validated eight-element absolute target.
            policy_state: Current measured pose and gripper closure.

        Returns:
            An independent absolute pose command for the next environment step.
        """
        return goal.copy()
