# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation and action contracts for VLM policies."""

import numpy as np
import torch
from abc import ABC, abstractmethod


class VLMObservationAdapter(ABC):
    """Select camera observations and prepare robot state for each environment."""

    camera_keys: tuple[str, ...] = ()
    """Keys in camera_obs, in the order sent to the model; no naming convention is required."""

    def __init__(self, image_max_edge: int = 384):
        self.image_max_edge = image_max_edge

    @abstractmethod
    def extract_proprioception(self, env, observation) -> list[dict]:
        """Return one JSON-serializable state dictionary per environment.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.

        Returns:
            State dictionaries in environment order, with units and frames defined by the adapter.
        """


class AgentActionAdapter(ABC):
    """Validate model responses and convert commands into environment actions."""

    response_schema: dict
    """JSON schema requested from the model; decode_command must also validate the response locally."""

    @abstractmethod
    def validate_environment(self, env) -> None:
        """Check that the environment accepts this adapter's actions.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    @abstractmethod
    def decode_command(self, payload: dict, proprioception: dict) -> np.ndarray:
        """Validate a response and return one command with adapter-defined dimensions and units.

        Args:
            payload: Parsed model response.
            proprioception: State sent with this inference request.

        Returns:
            A nonempty, finite one-dimensional command array.
        """

    def validate_command(self, command: np.ndarray) -> None:
        """Check that a decoded command is a nonempty, finite vector.

        Args:
            command: Decoded command to validate before execution.
        """
        assert command.ndim == 1 and command.size > 0, "Expected one command vector"
        assert np.isfinite(command).all(), "Decoded command must be finite"

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: Validated command for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)


class ChunkActionAdapter(AgentActionAdapter):
    """Decode fixed-length command sequences for AgentCommandedChunkPolicy.

    Implement decode_command to return (chunk_size, command_dim) and response_schema
    to describe that response. command_to_action converts one row at a time.
    """

    def __init__(self, chunk_size: int):
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size

    def validate_command(self, command: np.ndarray) -> None:
        """Check a decoded chunk before any of its commands are executed.

        Args:
            command: Decoded array shaped (chunk_size, command_dim).
        """
        assert (
            command.ndim == 2 and command.shape[0] == self.chunk_size and command.shape[1] > 0
        ), "Expected (chunk_size, command_dim) commands"
        assert np.isfinite(command).all(), "Decoded chunk must be finite"


class GoalActionAdapter(AgentActionAdapter):
    """Decode goals and provide closed-loop commands for AgentCommandedGoalPolicy.

    Goals are finite vectors with adapter-defined dimensions and units. A new goal
    is requested when goal_reached returns True or max_steps commands have executed.
    """

    def __init__(self, max_steps: int):
        assert type(max_steps) is int and max_steps > 0, "max_steps must be a positive integer"
        self.max_steps = max_steps

    @abstractmethod
    def goal_reached(self, goal: np.ndarray, proprioception: dict) -> bool:
        """Check goal completion against the current measured state.

        Args:
            goal: Validated goal for one environment.
            proprioception: Current state from the observation adapter.

        Returns:
            Whether to request a new goal before the next action.
        """

    @abstractmethod
    def compute_command(self, goal: np.ndarray, proprioception: dict) -> np.ndarray:
        """Compute one command toward the goal using current measured state.

        Args:
            goal: Validated goal for one environment.
            proprioception: Current state from the observation adapter.

        Returns:
            A finite command vector for command_to_action.
        """
