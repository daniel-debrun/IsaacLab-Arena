# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID controller compatibility and command validation."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter
from isaaclab_arena_vlm_agent.commands import Command, CommandType

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


class DroidActionAdapter(AgentActionAdapter):
    """Support unscaled absolute DROID arm commands and a zero-to-one binary gripper."""

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType):
        super().__init__(embodiment, command_type)
        self._command_dim = 8 if command_type == CommandType.EE_POSE else None

    def is_robot_action_space_supported(self, env) -> bool:
        """Check controller compatibility and resolve the command dimension before inference.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

        if self.command_type == CommandType.JOINT_POSITION:
            self._command_dim = None
        manager = env.unwrapped.action_manager
        if manager.active_terms != ["arm_action", "gripper_action"]:
            return False
        arm = manager.get_term("arm_action")
        gripper = manager.get_term("gripper_action")
        arm_type = (
            DifferentialInverseKinematicsAction if self.command_type == CommandType.EE_POSE else JointPositionAction
        )
        if not isinstance(arm, arm_type):
            return False
        if not (
            isinstance(gripper, BinaryJointPositionZeroToOneAction)
            and gripper.action_dim == 1
            and arm.cfg.asset_name == gripper.cfg.asset_name == self.embodiment.get_scene_key()
            and np.all(np.asarray(arm.cfg.scale) == 1.0)
            and arm.cfg.clip is None
            and gripper.cfg.clip is None
        ):
            return False
        cfg = arm.cfg
        if self.command_type == CommandType.EE_POSE:
            return bool(
                arm.action_dim == 7
                and cfg.body_name == self.embodiment.get_command_body_name()
                and cfg.controller.command_type == "pose"
                and not cfg.controller.use_relative_mode
                and cfg.body_offset is None
            )
        supported = bool(arm.action_dim > 0 and not cfg.use_default_offset and np.all(np.asarray(cfg.offset) == 0.0))
        self._command_dim = arm.action_dim + 1 if supported else None
        return supported

    def decode_command(self, response: Command, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Return command values in the adapter-defined order.

        Args:
            response: Validated response in the configured command space.
            arena_world: Live Arena scene queries; absolute commands need no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            One command vector in the configured coordinate space.
        """
        return np.asarray(response.values, dtype=np.float32)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Check dimensions, finite values, and constraints for the selected command type.

        Call is_robot_action_space_supported first to resolve joint-command dimensions.

        Args:
            command: Packed absolute command and gripper target.
        """
        if not (
            self._command_dim is not None
            and command.shape == (self._command_dim,)
            and np.isfinite(command).all()
            and command[-1] in (0.0, 1.0)
        ):
            return False
        if self.command_type == CommandType.EE_POSE:
            return bool(np.isclose(np.linalg.norm(command[3:7]), 1.0, atol=1e-3))
        return True
