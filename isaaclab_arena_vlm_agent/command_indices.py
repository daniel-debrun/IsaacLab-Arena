# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read absolute command indices and semantics from active action terms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseCommandIndices:
    """Identify an absolute root-relative XYZ/XYZW command and its controlled body."""

    columns: slice
    """Seven action columns containing root-relative XYZ followed by XYZW."""
    asset_name: str
    """Scene key of the controlled articulation."""
    body_name: str
    """Controlled body whose pose is measured."""


@dataclass(frozen=True)
class GripperCommandIndex:
    """Identify a binary command and its native open and close inputs."""

    column: int
    """Action column containing the binary input."""
    term_name: str
    """Active action term associated with the gripper."""
    open_value: float
    """Native input requesting an open gripper."""
    close_value: float
    """Native input requesting a closed gripper."""


@dataclass(frozen=True)
class AbsoluteCommandIndices:
    """Describe absolute pose, joint-position, and binary-gripper action columns."""

    action_dim: int
    """Total number of native action columns."""
    poses: tuple[PoseCommandIndices, ...]
    """Absolute end-effector pose commands in action order."""
    joints: tuple[slice, ...]
    """Absolute joint-position columns in controller joint order."""
    grippers: tuple[GripperCommandIndex, ...]
    """Binary gripper inputs in action order."""


def read_absolute_command_indices(env) -> AbsoluteCommandIndices | None:
    """Read supported absolute action columns, or return None for incompatible controllers.

    Args:
        env: Arena environment, optionally gym-wrapped, with initialized action terms.

    Returns:
        Command indices for unscaled absolute targets without offsets or clipping, or None.
    """
    from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

    manager = env.unwrapped.action_manager
    poses, joints, grippers = [], [], []
    first_column = 0
    for name, dimension in zip(manager.active_terms, manager.action_term_dim, strict=True):
        term = manager.get_term(name)
        cfg = term.cfg
        if dimension <= 0 or cfg.clip is not None:
            return None
        columns = slice(first_column, first_column + dimension)
        # Only known implementations are accepted: subclasses may change input semantics.
        if type(term) is DifferentialInverseKinematicsAction:
            if (
                dimension != 7
                or cfg.controller.command_type != "pose"
                or cfg.controller.use_relative_mode
                or not _all_equal(cfg.scale, 1.0)
                or cfg.body_offset is not None
            ):
                return None
            poses.append(PoseCommandIndices(columns, cfg.asset_name, cfg.body_name))
        elif type(term) is JointPositionAction:
            if cfg.use_default_offset or not _all_equal(cfg.scale, 1.0) or not _all_equal(cfg.offset, 0.0):
                return None
            joints.append(columns)
        elif type(term) in (BinaryJointPositionZeroToOneAction, BinaryJointPositionAction):
            if dimension != 1:
                return None
            values = (0.0, 1.0) if type(term) is BinaryJointPositionZeroToOneAction else (1.0, -1.0)
            grippers.append(GripperCommandIndex(first_column, name, *values))
        else:
            return None
        first_column += dimension
    if first_column == 0 or first_column != manager.total_action_dim:
        return None
    return AbsoluteCommandIndices(first_column, tuple(poses), tuple(joints), tuple(grippers))


def _all_equal(value, expected: float) -> bool:
    """Check scalar, per-axis, or regex-mapped controller scale and offset settings."""
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        return all(item == expected for item in value)
    return value == expected
