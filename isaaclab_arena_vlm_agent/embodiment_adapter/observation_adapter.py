# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for VLM policies."""

from abc import ABC, abstractmethod

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class VLMObservationAdapter(ABC):
    """Select camera observations and prepare state for VLM requests."""

    camera_keys: tuple[str, ...] = ()
    """Keys in camera_obs, in the order sent to the model; no naming convention is required."""

    @abstractmethod
    def extract_prompt_state(self, env, observation, env_id: int) -> dict:
        """Return only the measured state to include in one VLM request.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.
            env_id: Environment index to prepare the request for.

        Returns:
            A JSON-serializable dictionary of prompt fields, or an empty dictionary
            for image-only state input. Define field units and frames in the adapter.
        """
