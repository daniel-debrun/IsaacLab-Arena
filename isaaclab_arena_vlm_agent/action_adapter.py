# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured action adapters for step, chunk, and goal policies."""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.commands import (
    CommandType,
    JointPositionChunk,
    JointPositionCommand,
    PoseChunk,
    PoseCommand,
)
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


class AgentActionAdapter:
    """Convert explicitly typed absolute commands into an embodiment's native actions.

    The built-in formats support unscaled absolute arm actions and a zero-to-one binary gripper.
    Joint targets follow the arm action term's resolved joint order.
    """

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType):
        """Select the command schema independently of the robot controller.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
        """
        assert isinstance(command_type, CommandType), "command_type must be a CommandType"
        self.embodiment = embodiment
        """Robot defining the scene entity and, for pose commands, the command body."""
        self.command_type = command_type
        """Coordinate space of the decoded commands."""
        self.response_model = PoseCommand if command_type == CommandType.EE_POSE else JointPositionCommand
        self._command_dim = 8 if command_type == CommandType.EE_POSE else None

    def is_robot_action_space_supported(self, env) -> bool:
        """Check controller compatibility and resolve the command dimension before inference.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

        if self.command_type == CommandType.JOINT_POSITION:
            self._command_dim = None
        manager = env.unwrapped.action_manager
        if manager.active_terms != ["arm_action", "gripper_action"]:
            return False
        arm = manager.get_term("arm_action")
        gripper = manager.get_term("gripper_action")
        arm_type = (
            DifferentialInverseKinematicsAction if self.command_type == CommandType.EE_POSE else JointPositionAction
        )
        if not isinstance(arm, arm_type):
            return False
        if not (
            isinstance(gripper, BinaryJointPositionZeroToOneAction)
            and gripper.action_dim == 1
            and arm.cfg.asset_name == gripper.cfg.asset_name == self.embodiment.get_scene_key()
            and np.all(np.asarray(arm.cfg.scale) == 1.0)
            and arm.cfg.clip is None
            and gripper.cfg.clip is None
        ):
            return False
        cfg = arm.cfg
        if self.command_type == CommandType.EE_POSE:
            return bool(
                arm.action_dim == 7
                and cfg.body_name == self.embodiment.get_command_body_name()
                and cfg.controller.command_type == "pose"
                and not cfg.controller.use_relative_mode
                and cfg.body_offset is None
            )
        supported = bool(arm.action_dim > 0 and not cfg.use_default_offset and np.all(np.asarray(cfg.offset) == 0.0))
        self._command_dim = arm.action_dim + 1 if supported else None
        return supported

    def decode_command(
        self, response: PoseCommand | JointPositionCommand, arena_world: ArenaWorld, env_id: int
    ) -> np.ndarray:
        """Pack an absolute target followed by binary gripper closure.

        Args:
            response: Validated response in the configured command space.
            arena_world: Live Arena scene queries; absolute commands need no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            Pose or joint-position vector, followed by zero/open or one/closed.
        """
        if self.command_type == CommandType.EE_POSE:
            values = [*response.position_root_m, *response.quaternion_root_xyzw]
        else:
            values = response.joint_positions
        return np.asarray([*values, float(response.gripper_closed)], dtype=np.float32)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Check dimensions, finite values, and constraints for the selected command type.

        Call is_robot_action_space_supported first to resolve joint-command dimensions.

        Args:
            command: Packed absolute command and gripper target.
        """
        if not (
            self._command_dim is not None
            and command.shape == (self._command_dim,)
            and np.isfinite(command).all()
            and command[-1] in (0.0, 1.0)
        ):
            return False
        if self.command_type == CommandType.EE_POSE:
            return bool(np.isclose(np.linalg.norm(command[3:7]), 1.0, atol=1e-3))
        return True

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: One validated action row for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)


class ChunkActionAdapter(AgentActionAdapter):
    """Execute fixed-length command sequences in the selected coordinate space."""

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType, chunk_size: int = 1):
        """Bind fixed-length command chunks to an embodiment.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
            chunk_size: Number of commands in each prediction; defaults to one.
        """
        super().__init__(embodiment, command_type)
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size
        self.response_model = PoseChunk if command_type == CommandType.EE_POSE else JointPositionChunk

    def decode_command(
        self, response: PoseChunk | JointPositionChunk, arena_world: ArenaWorld, env_id: int
    ) -> np.ndarray:
        """Pack each validated command in execution order.

        Args:
            response: Validated sequence in the configured command space.
            arena_world: Live Arena scene queries; absolute commands need no pose offset.
            env_id: Environment index within the batched world state.

        Returns:
            Array of shape (number of commands, command_dim), checked before execution.
        """
        commands = []
        for command in response.commands:
            commands.append(super().decode_command(command, arena_world, env_id))
        if len({command.shape for command in commands}) != 1:
            raise InvalidAgentResponseError("All commands in a chunk must have the same dimensions")
        return np.stack(commands)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether every command is valid and the chunk has its configured length.

        Args:
            command: Packed command chunk.
        """
        return bool(
            command.ndim == 2
            and command.shape[0] == self.chunk_size
            and all(AgentActionAdapter.is_command_valid(self, row) for row in command)
        )


class GoalActionAdapter(AgentActionAdapter):
    """Decode goals in the coordinate space shared with a motion generator."""

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType, motion_generator: MotionGenerator):
        """Bind decoded goals to an embodiment and its motion generator.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
            motion_generator: Generator converting decoded goals into commands.
        """
        super().__init__(embodiment, command_type)
        assert isinstance(motion_generator, MotionGenerator)
        assert motion_generator.command_type == command_type, "Motion generator and adapter command types must match"
        self.motion_generator = motion_generator
        """Generate commands for this adapter's decoded goals."""
