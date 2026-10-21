# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Absolute DROID pose commands for single-step, chunk, and goal policies."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.droid_pose import DroidPoseChunk, DroidPoseCommand

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


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

    def decode_command(self, response: DroidPoseCommand, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Return the absolute target without applying the measured pose as an offset.

        Args:
            response: Validated target in meters, XYZW, and binary gripper closure.
            arena_world: Live Arena scene queries; absolute commands need no pose offset.
            env_id: Environment index within the batched world state.

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

    def decode_command(self, response: DroidPoseChunk, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Pack each validated pose in execution order.

        Args:
            response: Validated sequence of absolute poses and gripper targets.
            arena_world: Live Arena scene queries; absolute commands need no pose offset.
            env_id: Environment index within the batched world state.

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
    """Decode absolute DROID pose goals for the supplied motion generator."""
