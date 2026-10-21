# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Action contracts for VLM policies."""

import numpy as np
import torch
from abc import ABC, abstractmethod

from pydantic import BaseModel


class AgentActionAdapter(ABC):
    """Validate model responses and convert commands into robot actions."""

    response_model: type[BaseModel]
    """Response model used to generate the requested schema and validate model output."""

    @abstractmethod
    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the robot action space supports this adapter's commands.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    @abstractmethod
    def decode_command(self, response: BaseModel, policy_state: dict) -> np.ndarray:
        """Decode a validated response into commands with adapter-defined dimensions and units.

        Use InvalidAgentResponseError to request a correction for robot-specific constraints.

        Args:
            response: Model output validated against response_model.
            policy_state: Measured state sent with this inference request.

        Returns:
            A nonempty, finite command vector, or a chunk as defined by the adapter.
        """

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
    """Decode fixed-length command sequences for VLMChunkPolicy.

    Implement decode_command to return (chunk_size, command_dim) and response_model
    to describe that response. command_to_action converts one row at a time.
    """

    def __init__(self, chunk_size: int):
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a decoded chunk has the required shape and finite values.

        Args:
            command: Decoded array shaped (chunk_size, command_dim).
        """
        return bool(
            command.ndim == 2
            and command.shape[0] == self.chunk_size
            and command.shape[1] > 0
            and np.isfinite(command).all()
        )


class GoalActionAdapter(AgentActionAdapter):
    """Decode goals and provide closed-loop commands for VLMGoalPolicy.

    Goals are finite vectors with adapter-defined dimensions and units. A new goal
    is requested when goal_reached returns True or max_steps commands have executed.
    """

    def __init__(self, max_steps: int):
        assert type(max_steps) is int and max_steps > 0, "max_steps must be a positive integer"
        self.max_steps = max_steps

    @abstractmethod
    def goal_reached(self, goal: np.ndarray, policy_state: dict) -> bool:
        """Check goal completion against the current measured state.

        Args:
            goal: Validated goal for one environment.
            policy_state: Current state from the observation adapter.

        Returns:
            Whether to request a new goal before the next action.
        """

    @abstractmethod
    def compute_command(self, goal: np.ndarray, policy_state: dict) -> np.ndarray:
        """Compute one command toward the goal using current measured state.

        Args:
            goal: Validated goal for one environment.
            policy_state: Current state from the observation adapter.

        Returns:
            A finite command vector for command_to_action.
        """
