# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for VLM policies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class PolicyState(BaseModel):
    """Declare only the measured fields needed for inference and command execution."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


PolicyStateT = TypeVar("PolicyStateT", bound=PolicyState)


class VLMObservationAdapter(ABC, Generic[PolicyStateT]):
    """Select camera observations and prepare robot state for each environment."""

    camera_keys: tuple[str, ...] = ()
    """Keys in camera_obs, in the order sent to the model; no naming convention is required."""

    state_model: type[PolicyStateT]
    """Concrete state model shared with the action adapter; fields define units and frames."""

    @abstractmethod
    def extract_policy_state(self, env, observation) -> list[PolicyStateT]:
        """Return only the typed state needed for inference and command execution.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.

        Returns:
            Instances of state_model in environment order. Each model declares the exact
            measured fields, units, and frames needed by the paired action adapter and prompt.
        """
