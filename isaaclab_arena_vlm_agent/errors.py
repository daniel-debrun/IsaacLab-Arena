# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-response errors shared by VLM policies and embodiment adapters."""

from dataclasses import dataclass
from typing import Literal


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
