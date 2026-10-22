# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured action adapters for step, chunk, and goal policies."""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from isaaclab_arena_vlm_agent.commands import Command, CommandChunk, CommandType
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_world import ArenaWorld


class AgentActionAdapter(ABC):
    """Validate model responses and convert commands into robot actions."""

    response_model = Command

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType):
        """Select the command interpretation independently of the robot controller.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
        """
        assert isinstance(command_type, CommandType), "command_type must be a CommandType"
        self.embodiment = embodiment
        """Robot defining the scene entity and, for pose commands, the command body."""
        self.command_type = command_type
        """Coordinate space of the decoded commands."""

    @abstractmethod
    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the robot action space supports this adapter's commands.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    @abstractmethod
    def decode_command(self, response: BaseModel, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Decode a validated response into commands with adapter-defined dimensions and units.

        Use InvalidAgentResponseError to request correction of robot-specific constraints.

        Args:
            response: Model output validated against response_model.
            arena_world: Live Arena scene queries for command decoding.
            env_id: Environment index within the batched world state.

        Returns:
            A command vector, or a chunk as defined by the adapter.
        """

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a decoded command is a nonempty, finite vector.

        Args:
            command: Decoded command to validate before execution.
        """
        return bool(command.ndim == 1 and command.size > 0 and np.isfinite(command).all())

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: One validated action row for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)


class StepActionAdapter(AgentActionAdapter):
    """Use absolute native action terms in the embodiment's configured order.

    Pose terms use root-relative XYZ and XYZW. Joint terms use controller joint order.
    Binary terms retain their controller's native open/close values.
    """

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType):
        super().__init__(embodiment, command_type)
        self._command_dim = None
        self._quaternion_slices: list[slice] = []
        self._binary_values: dict[int, tuple[float, float]] = {}

    def is_robot_action_space_supported(self, env) -> bool:
        """Check absolute pose or joint controllers using the embodiment's action configuration.

        Term names, assets, body names, and joint counts come from the configuration.
        Unscaled, unoffset targets are required because commands pass directly to the controller.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """
        from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        self._command_dim = None
        self._quaternion_slices = []
        self._binary_values = {}
        action_cfg = self.embodiment.get_action_cfg()
        manager = env.unwrapped.action_manager
        dimension = 0
        has_target = False
        for name in manager.active_terms:
            term = manager.get_term(name)
            cfg = term.cfg
            expected = getattr(action_cfg, name, None)
            if expected is None or cfg.asset_name != expected.asset_name or cfg.clip is not None:
                return False
            if isinstance(term, BinaryJointPositionAction):
                values = getattr(term, "binary_action_values", None)
                if values is None and type(term).process_actions is BinaryJointPositionAction.process_actions:
                    values = (1.0, -1.0)
                if values is None or term.action_dim != 1:
                    return False
                self._binary_values[dimension] = values
            elif self.command_type == CommandType.EE_POSE and isinstance(term, DifferentialInverseKinematicsAction):
                if not (
                    term.action_dim == 7
                    and cfg.body_name == getattr(expected, "body_name", None)
                    and cfg.controller.command_type == "pose"
                    and not cfg.controller.use_relative_mode
                    and np.all(np.asarray(cfg.scale) == 1.0)
                    and cfg.body_offset is None
                ):
                    return False
                self._quaternion_slices.append(slice(dimension + 3, dimension + 7))
                has_target = True
            elif self.command_type == CommandType.JOINT_POSITION and isinstance(term, JointPositionAction):
                if not (
                    term.action_dim > 0
                    and not cfg.use_default_offset
                    and np.all(np.asarray(cfg.scale) == 1.0)
                    and np.all(np.asarray(cfg.offset) == 0.0)
                ):
                    return False
                has_target = True
            else:
                return False
            dimension += term.action_dim
        if has_target:
            self._command_dim = dimension
        return has_target

    def decode_command(self, response: Command, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Return absolute targets in native action-term order.

        Args:
            response: Validated command values.
            arena_world: Live Arena scene queries; absolute commands need no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            One command vector for the configured embodiment.
        """
        return np.asarray(response.values, dtype=np.float32)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Validate targets against the layout resolved by is_robot_action_space_supported.

        Args:
            command: One command in native action-term order.
        """
        if not (
            self._command_dim is not None
            and command.shape == (self._command_dim,)
            and super().is_command_valid(command)
        ):
            return False
        quaternions_valid = all(
            np.isclose(np.linalg.norm(command[quaternion_slice]), 1.0, atol=1e-3)
            for quaternion_slice in self._quaternion_slices
        )
        return quaternions_valid and all(command[index] in values for index, values in self._binary_values.items())


class ChunkActionAdapter(StepActionAdapter):
    """Decode fixed-length command sequences for the configured embodiment."""

    response_model = CommandChunk

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType, chunk_size: int = 1):
        """Configure native command chunks without a motion generator.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
            chunk_size: Number of commands per prediction; defaults to one.
        """
        super().__init__(embodiment, command_type)
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size

    def decode_command(self, response: CommandChunk, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Pack absolute commands into a chunk in execution order.

        Args:
            response: Validated sequence of command vectors.
            arena_world: Live Arena scene queries; absolute commands need no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            Array of shape (number of commands, command_dim), checked before execution.
        """
        commands = []
        for command in response.commands:
            commands.append(super().decode_command(command, arena_world, env_id))
        if len({command.shape for command in commands}) != 1:
            raise InvalidAgentResponseError("All commands in a chunk must have the same dimensions")
        return np.stack(commands)

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Validate the configured chunk length and each command's native layout.

        Args:
            command: Decoded command chunk.
        """
        is_command_valid = super().is_command_valid
        return bool(
            command.ndim == 2 and command.shape[0] == self.chunk_size and all(is_command_valid(row) for row in command)
        )


class GoalActionAdapter(StepActionAdapter):
    """Decode native goals for the configured embodiment and motion generator."""

    def __init__(self, embodiment: EmbodimentBase, command_type: CommandType, motion_generator: MotionGenerator):
        """Pair an embodiment's command layout with a compatible motion generator.

        Args:
            embodiment: Robot receiving the commands.
            command_type: Coordinate space of the policy output.
            motion_generator: Generator converting decoded goals into commands.
        """
        super().__init__(embodiment, command_type)
        assert isinstance(motion_generator, MotionGenerator)
        assert motion_generator.command_type == command_type, "Motion generator and adapter command types must match"
        self.motion_generator = motion_generator
        """Generate commands for this adapter's decoded goals."""
