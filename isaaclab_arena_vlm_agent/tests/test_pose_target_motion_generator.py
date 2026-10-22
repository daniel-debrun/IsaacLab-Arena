# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check pose tracking with embodiment-defined measured state."""

import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena.embodiments.common.command_type import CommandType
from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator


@pytest.mark.parametrize("binary_values", [(0.0, 1.0), (1.0, -1.0)])
def test_completion_checks_every_pose_and_unclamped_binary_state(binary_values):
    row = [0, 0, 0, 0, 0, 0, 1, binary_values[1]] * 2
    measured = torch.tensor([row, row], dtype=torch.float32)
    embodiment = SimpleNamespace(
        get_measured_pose_and_gripper_closure=Mock(return_value=measured),
        get_controller_command_layout=Mock(
            return_value=(CommandType.EE_POSE, (3, 11), {7: binary_values, 15: binary_values})
        ),
    )
    generator = PoseTargetMotionGenerator(embodiment)
    env = SimpleNamespace(unwrapped=SimpleNamespace(action_manager=SimpleNamespace(total_action_dim=16)))
    generator.configure(env)
    goal = np.asarray(row, dtype=np.float32)
    assert generator.goal_reached(goal, env, 0)
    embodiment.get_measured_pose_and_gripper_closure.assert_called_with(env)
    measured[1, 8] = 0.2  # Only the second pose in environment 1 misses its target.
    assert not generator.goal_reached(goal, env, 1)
    measured[1, 8] = 0
    measured[1, 11:15] *= -1  # Equivalent quaternion sign must not prevent completion.
    assert generator.goal_reached(goal, env, 1)
    # Overshoot stays visible regardless of the native binary input convention.
    measured[1, 15] += 0.2 * (binary_values[1] - binary_values[0])
    assert not generator.goal_reached(goal, env, 1)
    assert generator.goal_reached(goal, env, 0)
    command = generator.compute_command(goal, env, 0)
    np.testing.assert_array_equal(command, goal)
    assert not np.shares_memory(command, goal)


@pytest.mark.parametrize("invalid_state", [[0] * 7, [float("nan")] * 7])
def test_pose_only_layout_rejects_invalid_measured_rotation(invalid_state):
    measured = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    embodiment = SimpleNamespace(
        get_measured_pose_and_gripper_closure=Mock(return_value=measured),
        get_controller_command_layout=Mock(return_value=(CommandType.EE_POSE, (3,), {})),
    )
    generator = PoseTargetMotionGenerator(embodiment)
    env = SimpleNamespace(unwrapped=SimpleNamespace(action_manager=SimpleNamespace(total_action_dim=7)))
    generator.configure(env)
    goal = measured[0].numpy().copy()
    assert generator.goal_reached(goal, env, 0)
    measured[0] = torch.tensor(invalid_state)
    assert not generator.goal_reached(goal, env, 0)
