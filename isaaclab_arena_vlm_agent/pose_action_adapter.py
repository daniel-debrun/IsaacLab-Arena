# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured absolute pose commands for step, chunk, and goal policies."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.pose import PoseChunk, PoseCommand

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


def _pose_command(response: PoseCommand) -> np.ndarray:
    """Pack absolute position, XYZW rotation, and binary gripper closure."""
    return np.asarray(
        [*response.position_root_m, *response.quaternion_root_xyzw, float(response.gripper_closed)], dtype=np.float32
    )


class PoseActionAdapter(AgentActionAdapter):
    """Convert command-body targets into absolute IK actions for an embodiment."""

    response_model = PoseCommand

    def __init__(self, embodiment: EmbodimentBase):
        """Bind pose commands to an embodiment's scene entity and command body.

        Args:
            embodiment: Robot using absolute, unscaled IK and zero-to-one binary gripper actions.
        """
        self.embodiment = embodiment
        """Robot defining the scene key and command body for root-relative pose targets."""

    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the live actions match the embodiment and supported pose/gripper conventions.

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
            and cfg.asset_name == gripper.cfg.asset_name == self.embodiment.get_scene_key()
            and cfg.body_name == self.embodiment.get_command_body_name()
            and cfg.controller.command_type == "pose"
            and not cfg.controller.use_relative_mode
            and np.all(np.asarray(cfg.scale) == 1.0)
            and cfg.body_offset is None
            and cfg.clip is None
            and gripper.cfg.clip is None
        )

    def decode_command(self, response: PoseCommand, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
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


class PoseChunkActionAdapter(ChunkActionAdapter, PoseActionAdapter):
    """Execute absolute command-body poses, one per environment step."""

    response_model = PoseChunk

    def __init__(self, embodiment: EmbodimentBase, chunk_size: int):
        PoseActionAdapter.__init__(self, embodiment)
        ChunkActionAdapter.__init__(self, chunk_size)

    def decode_command(self, response: PoseChunk, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
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
            and all(PoseActionAdapter.is_command_valid(self, row) for row in command)
        )


class PoseGoalActionAdapter(GoalActionAdapter, PoseActionAdapter):
    """Decode absolute command-body goals for an embodiment and motion generator."""

    def __init__(self, embodiment: EmbodimentBase, motion_generator: MotionGenerator):
        PoseActionAdapter.__init__(self, embodiment)
        GoalActionAdapter.__init__(self, motion_generator)
