# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declared native command semantics and layout for an embodiment."""

from dataclasses import dataclass, field
from enum import Enum


class CommandType(str, Enum):
    """Coordinate space of absolute commands, independent of execution strategy."""

    EE_POSE = "ee_pose"
    """Command-body pose in the robot-root frame, in meters and XYZW quaternions."""

    JOINT_POSITION = "joint_position"
    """Joint positions in controller order, in radians or meters for prismatic joints."""


@dataclass
class CommandSpace:
    """Describe controller inputs after controller configuration is finalized.

    Declaring an absolute command type promises native, unscaled targets without offsets.
    Update this declaration when changing controller semantics or action layout.
    """

    command_type: CommandType
    """Coordinate space accepted by the configured controller."""

    dimension: int
    """Number of targets per environment, including auxiliary controls."""

    quaternion_offsets: tuple[int, ...] = ()
    """Start indices of unit XYZW quaternions in the command vector."""

    binary_values: dict[int, tuple[float, float]] = field(default_factory=dict)
    """Allowed values for binary controls, keyed by command-vector index."""
