# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID base_link poses in the robot-root frame, in meters and XYZW quaternions."""

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import PolicyState

Position = Annotated[list[float], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[float], Field(min_length=4, max_length=4)]


def _unit_quaternion(value: list[float]) -> list[float]:
    """Validate a unit XYZW quaternion without changing its rotation."""
    if not math.isclose(math.hypot(*value), 1.0, abs_tol=1e-3):
        raise ValueError("quaternion_root_xyzw must have unit length")
    return value


class DroidEndEffectorPoseAndGripperState(PolicyState):
    """Measured T_R_E and gripper closure; R is robot root and E is Robotiq base_link."""

    position_root_m: Position
    """Translation t_R_E in meters, excluding grasp and controller offsets."""

    quaternion_root_xyzw: Quaternion
    """Rotation q_R_E as a unit quaternion in XYZW order."""

    gripper_closed_fraction: float
    """Measured finger position: zero is open, one is closed; physical overshoot is preserved."""

    _validate_quaternion = field_validator("quaternion_root_xyzw")(_unit_quaternion)


class DroidPoseCommand(BaseModel):
    """Request absolute T_R_E and a binary gripper target."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    position_root_m: Position
    """Target translation t_R_E in meters for Robotiq base_link relative to robot root."""

    quaternion_root_xyzw: Quaternion
    """Target rotation q_R_E as a unit quaternion in XYZW order."""

    gripper_closed: bool
    """True closes the gripper; False opens it."""

    _validate_quaternion = field_validator("quaternion_root_xyzw")(_unit_quaternion)


class DroidPoseChunk(BaseModel):
    """Ordered absolute pose commands, one per environment step."""

    model_config = ConfigDict(extra="forbid", strict=True)

    commands: list[DroidPoseCommand] = Field(min_length=1)
    """Commands in execution order; the action adapter enforces its configured chunk length."""
