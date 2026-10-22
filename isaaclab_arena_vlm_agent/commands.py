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


class PoseCommand(BaseModel):
    """Request absolute T_R_E (robot root R, embodiment command body E) and a binary gripper target."""

    position_root_m: list[float] = Field(min_length=3, max_length=3)
    """Target translation t_R_E in meters for the embodiment command body relative to robot root."""

    quaternion_root_xyzw: list[float] = Field(min_length=4, max_length=4)
    """Target rotation q_R_E as a unit quaternion in XYZW order."""

    gripper_closed: bool
    """True closes the gripper; False opens it."""


class PoseChunk(BaseModel):
    """Ordered absolute pose commands, one per environment step."""

    commands: list[PoseCommand] = Field(min_length=1)
    """Commands in execution order; the action adapter enforces its configured chunk length."""


class JointPositionCommand(BaseModel):
    """Request absolute arm joint positions and a binary gripper target."""

    joint_positions: list[float] = Field(min_length=1)
    """Targets in the arm action term's resolved joint order, in radians or meters."""

    gripper_closed: bool
    """True closes the gripper; False opens it."""


class JointPositionChunk(BaseModel):
    """Ordered absolute joint-position commands, one per environment step."""

    commands: list[JointPositionCommand] = Field(min_length=1)
    """Commands in execution order; the action adapter enforces its configured chunk length."""
