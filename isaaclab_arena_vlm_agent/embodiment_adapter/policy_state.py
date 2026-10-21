# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed state shared by VLM observation and action adapters."""

from typing import TypeVar

from pydantic import BaseModel, ConfigDict


class PolicyState(BaseModel):
    """Declare only the measured fields needed for inference and command execution."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


PolicyStateT = TypeVar("PolicyStateT", bound=PolicyState)
