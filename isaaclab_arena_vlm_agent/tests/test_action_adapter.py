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
    return SimpleNamespace(get_controller_command_layout=Mock())


@pytest.mark.parametrize("layout", [None, (CommandType.JOINT_POSITION, (), {})])
def test_unsupported_controllers_clear_previous_validation(layout, embodiment):
    adapter = StepActionAdapter(embodiment, CommandType.EE_POSE)
    embodiment.get_controller_command_layout.return_value = (CommandType.EE_POSE, (3,), {7: (0.0, 1.0)})
    assert adapter.is_robot_action_space_supported(_env(8))
    embodiment.get_controller_command_layout.return_value = layout
    assert not adapter.is_robot_action_space_supported(_env(8))
    assert not adapter.is_command_valid(np.ones(8))


@pytest.mark.parametrize("dimension", [7, 29])
def test_joint_commands_use_controller_dimension_without_gripper(dimension, embodiment):
    embodiment.get_controller_command_layout.return_value = (CommandType.JOINT_POSITION, (), {})
    adapter = ChunkActionAdapter(embodiment, CommandType.JOINT_POSITION, chunk_size=2)
    assert adapter.is_robot_action_space_supported(_env(dimension))
    assert adapter.is_command_valid(np.zeros((2, dimension)))
    assert not adapter.is_command_valid(np.zeros((2, dimension + 1)))
    assert not adapter.is_command_valid(np.full((2, dimension), np.nan))


@pytest.mark.parametrize("binary_values", [(0.0, 1.0), (1.0, -1.0)])
def test_bimanual_pose_layout_and_binary_conventions(binary_values, embodiment):
    embodiment.get_controller_command_layout.return_value = (
        CommandType.EE_POSE,
        (3, 11),
        {7: binary_values, 15: binary_values},
    )
    adapter = ChunkActionAdapter(embodiment, CommandType.EE_POSE, chunk_size=2)
    env = _env(16)
    assert adapter.is_robot_action_space_supported(env)
    embodiment.get_controller_command_layout.assert_called_once_with(env)
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
