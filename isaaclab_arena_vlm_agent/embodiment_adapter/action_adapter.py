# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Action contracts for VLM policies."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal

from pydantic import BaseModel

from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import PolicyStateT


@dataclass(frozen=True)
class AgentResponseErrorDetails:
    """Describe a rejected model response with a stable code and correction message."""

    code: Literal["invalid_schema", "invalid_command"]
    message: str


class InvalidAgentResponseError(ValueError):
    """A model response was rejected and may be corrected on another inference attempt."""

    def __init__(self, message: str, *, code: Literal["invalid_schema", "invalid_command"] = "invalid_command"):
        super().__init__(message)
        self.details = AgentResponseErrorDetails(code=code, message=message)


class AgentActionAdapter(ABC, Generic[PolicyStateT]):
    """Validate model responses and convert commands into robot actions."""

    state_model: type[PolicyStateT]
    """Concrete measured-state model required for decoding responses into actions."""

    response_model: type[BaseModel]
    """Response model used to generate the requested schema and validate model output."""

    @abstractmethod
    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the robot action space supports this adapter's commands.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    @abstractmethod
    def decode_command(self, response: BaseModel, policy_state: PolicyStateT) -> np.ndarray:
        """Decode a validated response into commands with adapter-defined dimensions and units.

        Use InvalidAgentResponseError to request a correction for robot-specific constraints.

        Args:
            response: Model output validated against response_model.
            policy_state: Typed state sent with this inference request.

        Returns:
            A nonempty, finite array shaped (horizon, action_dim), executed in order.
            A single-step response has horizon 1; targets must be decoded into action rows.
        """

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether decoded actions form a nonempty, finite (horizon, action_dim) array.

        Args:
            command: Decoded command to validate before execution.
        """
        return bool(command.ndim == 2 and command.shape[0] > 0 and command.shape[1] > 0 and np.isfinite(command).all())

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: One validated action row for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)
