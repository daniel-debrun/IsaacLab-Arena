# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate declared command spaces without runtime action terms."""

import numpy as np
from types import SimpleNamespace

import pytest

from isaaclab_arena.embodiments.common.command_space import CommandSpace, CommandType
from isaaclab_arena_vlm_agent.action_adapter import ChunkActionAdapter, StepActionAdapter


def _env(dimension):
    # No action manager is needed to check the command contract.
    return SimpleNamespace(unwrapped=SimpleNamespace(single_action_space=SimpleNamespace(shape=(dimension,))))


@pytest.mark.parametrize(
    "space",
    [None, CommandSpace(CommandType.JOINT_POSITION, 8), CommandSpace(CommandType.EE_POSE, 7)],
)
def test_unsupported_command_space_clears_previous_validation(space):
    embodiment = SimpleNamespace(command_space=CommandSpace(CommandType.EE_POSE, 8))
    adapter = StepActionAdapter(embodiment, CommandType.EE_POSE)
    assert adapter.is_robot_action_space_supported(_env(8))
    embodiment.command_space = space
    assert not adapter.is_robot_action_space_supported(_env(8))
    assert not adapter.is_command_valid(np.ones(8))


@pytest.mark.parametrize("dimension", [7, 29])
def test_joint_commands_use_declared_dimension_without_gripper(dimension):
    embodiment = SimpleNamespace(command_space=CommandSpace(CommandType.JOINT_POSITION, dimension))
    adapter = ChunkActionAdapter(embodiment, CommandType.JOINT_POSITION, chunk_size=2)
    assert adapter.is_robot_action_space_supported(_env(dimension))
    assert adapter.is_command_valid(np.zeros((2, dimension)))
    assert not adapter.is_command_valid(np.zeros((2, dimension + 1)))
    assert not adapter.is_command_valid(np.full((2, dimension), np.nan))


@pytest.mark.parametrize("binary_values", [(0.0, 1.0), (1.0, -1.0)])
def test_bimanual_pose_layout_and_binary_conventions(binary_values):
    embodiment = SimpleNamespace(
        command_space=CommandSpace(
            CommandType.EE_POSE,
            16,
            quaternion_offsets=(3, 11),
            binary_values={7: binary_values, 15: binary_values},
        )
    )
    adapter = ChunkActionAdapter(embodiment, CommandType.EE_POSE, chunk_size=2)
    assert adapter.is_robot_action_space_supported(_env(16))
    row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, binary_values[0]] * 2
    response = adapter.response_model.model_validate({"commands": [{"values": row}] * 2})
    commands = adapter.decode_command(response, None, 0)
    assert adapter.is_command_valid(commands)
    commands[1, 11:15] = 0.0
    assert not adapter.is_command_valid(commands)
    commands[1] = row
    commands[0, 7] = 0.5
    assert not adapter.is_command_valid(commands)
    commands[0, 7] = binary_values[1]
    assert adapter.is_command_valid(commands)
