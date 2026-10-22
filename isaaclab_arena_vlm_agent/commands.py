# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLM command types and response schemas."""

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    """Coordinate space of absolute commands, independent of execution strategy."""

    EE_POSE = "ee_pose"
    """Command-body pose in the robot-root frame, in meters and XYZW quaternions."""

    JOINT_POSITION = "joint_position"
    """Joint positions in controller order, in radians or meters for prismatic joints."""


class Command(BaseModel):
    """Request one command in the adapter-defined layout."""

    values: list[float] = Field(min_length=1)
    """Ordered targets; the adapter defines dimensions, units, and interpretation for the embodiment."""


CommandT = TypeVar("CommandT", bound=BaseModel)


class CommandChunk(BaseModel, Generic[CommandT]):
    """Ordered commands, one per environment step."""

    commands: list[CommandT] = Field(min_length=1)
    """The action adapter validates each target and enforces the configured chunk length."""
