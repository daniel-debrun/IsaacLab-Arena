# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-body poses in the robot-root frame, in meters and XYZW quaternions."""

from pydantic import BaseModel, Field


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
