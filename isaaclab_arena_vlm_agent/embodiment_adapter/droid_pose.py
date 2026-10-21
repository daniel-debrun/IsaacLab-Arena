# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID base_link poses in the robot-root frame, in meters and XYZW quaternions."""

from pydantic import BaseModel, Field


class DroidPoseCommand(BaseModel):
    """Request absolute T_R_E (robot root R, Robotiq base_link E) and a binary gripper target."""

    position_root_m: list[float] = Field(min_length=3, max_length=3)
    """Target translation t_R_E in meters for Robotiq base_link relative to robot root."""

    quaternion_root_xyzw: list[float] = Field(min_length=4, max_length=4)
    """Target rotation q_R_E as a unit quaternion in XYZW order."""

    gripper_closed: bool
    """True closes the gripper; False opens it."""


class DroidPoseChunk(BaseModel):
    """Ordered absolute pose commands, one per environment step."""

    commands: list[DroidPoseCommand] = Field(min_length=1)
    """Commands in execution order; the action adapter enforces its configured chunk length."""
