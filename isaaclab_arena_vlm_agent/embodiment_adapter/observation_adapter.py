# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for VLM policies."""

from abc import ABC, abstractmethod

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class VLMObservationAdapter(ABC):
    """Select camera observations and prepare robot state for each environment."""

    camera_keys: tuple[str, ...] = ()
    """Keys in camera_obs, in the order sent to the model; no naming convention is required."""

    @abstractmethod
    def extract_proprioception(self, env, observation) -> list[dict]:
        """Return one JSON-serializable state dictionary per environment.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.

        Returns:
            State dictionaries in environment order, with units and frames defined by the adapter.
        """
