# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete adapters for controllers accepting absolute targets directly."""

import numpy as np

from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.command_indices import read_absolute_command_indices
from isaaclab_arena_vlm_agent.commands import Command, CommandChunk
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError


class AbsoluteStepActionAdapter(AgentActionAdapter):
    """Decode absolute pose or joint targets and native binary gripper values."""

    response_model = Command

    def __init__(self):
        self._command_indices = None

    def is_robot_action_space_supported(self, env) -> bool:
        """Resolve command indices from the active controller before policy inference."""
        self._command_indices = read_absolute_command_indices(env)
        return self._command_indices is not None

    def decode_command(self, response: Command, arena_world, env_id: int) -> np.ndarray:
        """Decode native targets, requesting corrections for invalid dimensions or values.

        Args:
            response: Validated response containing targets in controller order.
            arena_world: Live scene queries; absolute targets require no measured offset.
            env_id: Environment index within the batched world state.

        Returns:
            A finite vector with unit XYZW quaternions and native binary values.
        """
        indices = self._command_indices
        assert indices is not None, "Check controller compatibility before decoding absolute commands"
        command = np.asarray(response.values, dtype=np.float32)
        if command.shape != (indices.action_dim,):
            raise InvalidAgentResponseError(f"Expected {indices.action_dim} values, got {command.size}")
        if not np.isfinite(command).all():
            raise InvalidAgentResponseError("All command values must be finite")
        for pose in indices.poses:
            start, stop = pose.columns.start + 3, pose.columns.stop
            if not np.isclose(np.linalg.norm(command[start:stop]), 1.0, atol=1e-3):
                raise InvalidAgentResponseError(f"Values {start} to {stop - 1} must form a unit XYZW quaternion")
        for gripper in indices.grippers:
            if command[gripper.column] not in (gripper.open_value, gripper.close_value):
                raise InvalidAgentResponseError(
                    f"Value {gripper.column} must be {gripper.open_value} (open) or {gripper.close_value} (closed)"
                )
        return command


class AbsoluteChunkActionAdapter(AbsoluteStepActionAdapter, ChunkActionAdapter):
    """Decode a required number of absolute commands for chunk execution."""

    response_model = CommandChunk

    def __init__(self, chunk_size: int):
        AbsoluteStepActionAdapter.__init__(self)
        ChunkActionAdapter.__init__(self, chunk_size)

    def decode_command(self, response: CommandChunk, arena_world, env_id: int) -> np.ndarray:
        """Validate and stack absolute commands in execution order."""
        if len(response.commands) != self.chunk_size:
            raise InvalidAgentResponseError(f"Expected {self.chunk_size} commands, got {len(response.commands)}")
        commands = []
        for command in response.commands:
            commands.append(super().decode_command(command, arena_world, env_id))
        return np.stack(commands)


class AbsoluteGoalActionAdapter(AbsoluteStepActionAdapter, GoalActionAdapter):
    """Decode absolute targets for an explicitly supplied motion generator."""

    def __init__(self, motion_generator: MotionGenerator):
        AbsoluteStepActionAdapter.__init__(self)
        GoalActionAdapter.__init__(self, motion_generator)
