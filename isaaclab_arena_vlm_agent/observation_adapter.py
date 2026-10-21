# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for VLM policies."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class VLMObservationAdapter(ABC):
    """Select observation modalities for VLM input independently of the chat format."""

    @abstractmethod
    def extract_vlm_input_from_observation(
        self, env, observation, env_id: int
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Extract camera images and prompt values for one environment using modality-specific helpers.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched environment observations.
            env_id: Environment index to prepare the request for.

        Returns:
            HWC uint8 RGB or RGBA images indexed by camera name in request order, and
            JSON-serializable prompt values. The policy handles image encoding and message assembly.
        """
