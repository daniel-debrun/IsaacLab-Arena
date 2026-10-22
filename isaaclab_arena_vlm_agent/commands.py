# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLM command types and response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    """Coordinate space of absolute commands, independent of execution strategy."""

    EE_POSE = "ee_pose"
    """Command-body pose in the robot-root frame, in meters and XYZW quaternions."""

    JOINT_POSITION = "joint_position"
    """Joint positions in controller order, in radians or meters for prismatic joints."""


class Command(BaseModel):
    """Request an absolute arm target and binary gripper state."""

    arm_target: list[float] = Field(min_length=1)
    """EE_POSE: root-relative [x, y, z, qx, qy, qz, qw]; JOINT_POSITION: joints in controller order."""

    gripper_closed: bool
    """True closes the gripper; False opens it."""


class CommandChunk(BaseModel):
    """Ordered commands, one per environment step."""

    commands: list[Command] = Field(min_length=1)
    """The action adapter validates each target and enforces the configured chunk length."""
