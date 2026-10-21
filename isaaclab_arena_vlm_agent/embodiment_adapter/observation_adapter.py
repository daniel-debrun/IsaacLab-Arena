# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for VLM policies."""

from abc import ABC, abstractmethod

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class VLMObservationAdapter(ABC):
    """Assemble selected observation modalities into VLM request content."""

    @abstractmethod
    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Build image and text content for one environment using modality-specific helpers.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.
            env_id: Environment index to prepare the request for.
            image_max_width_or_height: Maximum encoded image width or height, in pixels.

        Returns:
            Image and text inputs for the selected observations.
        """
