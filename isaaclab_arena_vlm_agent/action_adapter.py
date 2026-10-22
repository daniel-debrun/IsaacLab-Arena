# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured action adapters for step, chunk, and goal policies."""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.commands import Command, CommandChunk, CommandType
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


class AgentActionAdapter(ABC):
    """Validate model responses and convert commands into robot actions."""

    response_model = Command

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType):
        """Select the command interpretation independently of the robot controller.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
        """
        assert isinstance(command_type, CommandType), "command_type must be a CommandType"
        self.embodiment = embodiment
        """Robot defining the scene entity and, for pose commands, the command body."""
        self.command_type = command_type
        """Coordinate space of the decoded commands."""

    @abstractmethod
    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the robot action space supports this adapter's commands.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    def decode_command(self, response: Command, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Return command values in the adapter-defined order.

        Args:
            response: Validated response in the configured command space.
            arena_world: Live Arena scene queries; absolute commands need no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            One command vector in the configured coordinate space.
        """
        return np.asarray(response.values, dtype=np.float32)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a decoded command is a nonempty, finite vector.

        Args:
            command: Decoded command to validate before execution.
        """
        return bool(command.ndim == 1 and command.size > 0 and np.isfinite(command).all())

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

    response_model = CommandChunk

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

    def decode_command(self, response: CommandChunk, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
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
        is_command_valid = super().is_command_valid
        return bool(
            command.ndim == 2 and command.shape[0] == self.chunk_size and all(is_command_valid(row) for row in command)
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
