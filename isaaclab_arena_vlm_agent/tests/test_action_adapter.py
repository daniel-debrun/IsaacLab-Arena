# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate commands using normalized controller layouts."""

import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena.embodiments.common.command_type import CommandType
from isaaclab_arena_vlm_agent.action_adapter import ChunkActionAdapter, StepActionAdapter


def _env(dimension):
    return SimpleNamespace(unwrapped=SimpleNamespace(action_manager=SimpleNamespace(total_action_dim=dimension)))


@pytest.fixture
def embodiment():
    return SimpleNamespace(
        get_command_type=Mock(return_value=CommandType.ABS_EE_POSE),
        get_quaternion_indices=Mock(return_value=(3,)),
        get_binary_action_values=Mock(return_value={7: (0.0, 1.0)}),
    )


@pytest.mark.parametrize("command_type", [None, CommandType.ABS_JOINT_POSITION])
def test_unsupported_controllers_clear_previous_validation(command_type, embodiment):
    adapter = StepActionAdapter(embodiment, CommandType.ABS_EE_POSE)
    assert adapter.is_robot_action_space_supported(_env(8))
    embodiment.get_command_type.return_value = command_type
    assert not adapter.is_robot_action_space_supported(_env(8))
    assert not adapter.is_command_valid(np.ones(8))


@pytest.mark.parametrize("dimension", [7, 29])
def test_joint_commands_use_controller_dimension_without_gripper(dimension, embodiment):
    embodiment.get_command_type.return_value = CommandType.ABS_JOINT_POSITION
    embodiment.get_quaternion_indices.return_value = ()
    embodiment.get_binary_action_values.return_value = {}
    adapter = ChunkActionAdapter(embodiment, CommandType.ABS_JOINT_POSITION, chunk_size=2)
    assert adapter.is_robot_action_space_supported(_env(dimension))
    assert adapter.is_command_valid(np.zeros((2, dimension)))
    assert not adapter.is_command_valid(np.zeros((2, dimension + 1)))
    assert not adapter.is_command_valid(np.full((2, dimension), np.nan))


@pytest.mark.parametrize("binary_values", [(0.0, 1.0), (1.0, -1.0)])
def test_bimanual_pose_layout_and_binary_conventions(binary_values, embodiment):
    embodiment.get_quaternion_indices.return_value = (3, 11)
    embodiment.get_binary_action_values.return_value = {7: binary_values, 15: binary_values}
    adapter = ChunkActionAdapter(embodiment, CommandType.ABS_EE_POSE, chunk_size=2)
    env = _env(16)
    assert adapter.is_robot_action_space_supported(env)
    embodiment.get_command_type.assert_called_once_with()
    embodiment.get_quaternion_indices.assert_called_once_with()
    embodiment.get_binary_action_values.assert_called_once_with(env)
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
