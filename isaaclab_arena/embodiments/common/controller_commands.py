# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize active controller inputs for absolute robot commands."""

import numpy as np

from isaaclab_arena.embodiments.common.command_type import CommandType


def get_controller_command_layout(
    action_manager,
) -> tuple[CommandType, tuple[int, ...], dict[int, tuple[float, float]]] | None:
    """Derive the command type, quaternion offsets, and binary input values in action order.

    Only unscaled absolute pose or joint-position targets are supported. Relative,
    mixed pose/joint, offset, clipped, and unknown controllers return None.

    Args:
        action_manager: Initialized Isaac Lab action manager supplying resolved dimensions.
    """
    from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    command_type = None
    quaternion_offsets = []
    binary_values = {}
    offset = 0
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        cfg = term.cfg
        if cfg.clip is not None or term.action_dim <= 0:
            return None
        if isinstance(term, BinaryJointPositionAction):
            values = getattr(term, "binary_action_values", None)
            if values is None and type(term).process_actions is BinaryJointPositionAction.process_actions:
                values = (1.0, -1.0)
            if values is None or term.action_dim != 1:
                return None
            binary_values[offset] = values
        elif not isinstance(term, (DifferentialInverseKinematicsAction, JointPositionAction)):
            return None
        else:
            if not _all_equal(cfg.scale, 1.0):
                return None
            if isinstance(term, DifferentialInverseKinematicsAction):
                if (
                    cfg.controller.command_type != "pose"
                    or cfg.controller.use_relative_mode
                    or cfg.body_offset is not None
                    or term.action_dim != 7
                ):
                    return None
                term_type = CommandType.EE_POSE
                quaternion_offsets.append(offset + 3)
            else:
                if cfg.use_default_offset or not _all_equal(cfg.offset, 0.0):
                    return None
                term_type = CommandType.JOINT_POSITION
            if command_type is not None and command_type != term_type:
                return None
            command_type = term_type
        offset += term.action_dim
    if command_type is None:
        return None
    return command_type, tuple(quaternion_offsets), binary_values


def _all_equal(value, target: float) -> bool:
    """Check scalar, sequence, or joint-name mapping configuration values."""
    if isinstance(value, dict):
        value = list(value.values())
    return bool(np.all(np.asarray(value) == target))
