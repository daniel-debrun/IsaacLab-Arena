# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation and action contracts for a stateless VLM policy."""

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

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: Validated command for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)
