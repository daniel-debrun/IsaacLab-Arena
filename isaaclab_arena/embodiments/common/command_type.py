# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coordinate spaces used by robot commands."""

from enum import Enum


class CommandType(str, Enum):
    """Coordinate space of absolute commands, independent of execution strategy."""

    ABS_EE_POSE = "abs_ee_pose"
    """Absolute command-body pose in the robot-root frame, in meters and XYZW quaternions."""

    ABS_JOINT_POSITION = "abs_joint_position"
    """Absolute joint positions in controller order, in radians or meters for prismatic joints."""
