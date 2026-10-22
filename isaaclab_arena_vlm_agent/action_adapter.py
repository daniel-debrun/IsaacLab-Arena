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

from pydantic import BaseModel

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

    def __init__(self, action_adapter: AgentActionAdapter, chunk_size: int = 1):
        """Wrap a single-command adapter with fixed-length chunk decoding.

        Args:
            action_adapter: Adapter defining each command's schema, validation, and conversion.
            chunk_size: Number of commands in each prediction; defaults to one.
        """
        super().__init__(action_adapter.embodiment, action_adapter.command_type)
        self.action_adapter = action_adapter
        """Robot-specific handling shared by all commands in a chunk."""
        self.response_model = CommandChunk[action_adapter.response_model]
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size

    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the wrapped adapter supports the environment's actions."""
        return self.action_adapter.is_robot_action_space_supported(env)

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert one row using the wrapped adapter's native action convention."""
        return self.action_adapter.command_to_action(env, command)

    def decode_command(self, response: CommandChunk, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Pack each validated command in execution order.

        Args:
            response: Validated sequence in the configured command space.
            arena_world: Live Arena scene queries for per-command decoding.
            env_id: Environment index within the batched world state.

        Returns:
            Array of shape (number of commands, command_dim), checked before execution.
        """
        commands = []
        for command in response.commands:
            commands.append(self.action_adapter.decode_command(command, arena_world, env_id))
        if len({command.shape for command in commands}) != 1:
            raise InvalidAgentResponseError("All commands in a chunk must have the same dimensions")
        return np.stack(commands)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether every command is valid and the chunk has its configured length.

        Args:
            command: Packed command chunk.
        """
        is_command_valid = self.action_adapter.is_command_valid
        return bool(
            command.ndim == 2 and command.shape[0] == self.chunk_size and all(is_command_valid(row) for row in command)
        )


class GoalActionAdapter(AgentActionAdapter):
    """Decode goals in the coordinate space shared with a motion generator."""

    def __init__(self, action_adapter: AgentActionAdapter, motion_generator: MotionGenerator):
        """Pair a single-command adapter with a compatible motion generator.

        Args:
            action_adapter: Adapter defining the goal schema, validation, and action conversion.
            motion_generator: Generator converting decoded goals into commands.
        """
        super().__init__(action_adapter.embodiment, action_adapter.command_type)
        assert isinstance(motion_generator, MotionGenerator)
        assert (
            motion_generator.command_type == action_adapter.command_type
        ), "Motion generator and adapter command types must match"
        self.action_adapter = action_adapter
        """Robot-specific handling for goals and generated commands."""
        self.response_model = action_adapter.response_model
        self.motion_generator = motion_generator
        """Generate commands for this adapter's decoded goals."""

    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the wrapped adapter supports the environment's actions."""
        return self.action_adapter.is_robot_action_space_supported(env)

    def decode_command(self, response: BaseModel, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Decode a goal using the wrapped adapter."""
        return self.action_adapter.decode_command(response, arena_world, env_id)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Validate a goal using the wrapped adapter's command constraints."""
        return self.action_adapter.is_command_valid(command)

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a generated command using the wrapped adapter's native action convention."""
        return self.action_adapter.command_to_action(env, command)
