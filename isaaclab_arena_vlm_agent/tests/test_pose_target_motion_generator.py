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
@pytest.mark.parametrize("joint_targets", [(0.1, 0.461), (0.04, 0.0)])
def test_completion_checks_every_pose_and_unclamped_binary_state(binary_values, joint_targets):
    row = [0, 0, 0, 0, 0, 0, 1, binary_values[1]] * 2
    measured = torch.tensor([row, row], dtype=torch.float32)

    def get_measured_joint_position(env, name):
        native_value = measured[:, {"left_gripper": 7, "right_gripper": 15}[name]]
        closure = (native_value - binary_values[0]) / (binary_values[1] - binary_values[0])
        return joint_targets[0] + closure * (joint_targets[1] - joint_targets[0])

    embodiment = SimpleNamespace(
        get_measured_command_body_pose_in_root=Mock(
            side_effect=lambda env: torch.stack((measured[:, :7], measured[:, 8:15]), dim=1)
        ),
        get_gripper=Mock(
            return_value=SimpleNamespace(
                get_measured_joint_position=get_measured_joint_position,
                get_open_close_joint_positions=Mock(return_value=joint_targets),
            )
        ),
        get_command_type=Mock(return_value=CommandType.ABS_EE_POSE),
        get_quaternion_indices=Mock(return_value=(3, 11)),
        get_binary_action_values=Mock(return_value={7: binary_values, 15: binary_values}),
    )
    generator = PoseTargetMotionGenerator(embodiment)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_manager=SimpleNamespace(
                total_action_dim=16,
                active_terms=["left_arm", "left_gripper", "right_arm", "right_gripper"],
                action_term_dim=[7, 1, 7, 1],
            )
        )
    )
    goal = np.asarray(row, dtype=np.float32)
    assert generator.is_goal_reached(goal, env, 0)
    embodiment.get_measured_command_body_pose_in_root.assert_called_with(env)
    measured[1, 8] = 0.2  # Only the second pose in environment 1 misses its target.
    assert not generator.is_goal_reached(goal, env, 1)
    measured[1, 8] = 0
    measured[1, 11:15] *= -1  # Equivalent quaternion sign must not prevent completion.
    assert generator.is_goal_reached(goal, env, 1)
    # Overshoot stays visible regardless of the native binary input convention.
    measured[1, 15] += 0.2 * (binary_values[1] - binary_values[0])
    assert not generator.is_goal_reached(goal, env, 1)
    assert generator.is_goal_reached(goal, env, 0)
    command = generator.compute_command(goal, env, 0)
    np.testing.assert_array_equal(command, goal)
    assert not np.shares_memory(command, goal)


@pytest.mark.parametrize("invalid_state", [[0] * 7, [float("nan")] * 7])
def test_pose_only_layout_rejects_invalid_measured_rotation(invalid_state):
    measured = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    embodiment = SimpleNamespace(
        get_measured_command_body_pose_in_root=Mock(return_value=measured),
        get_command_type=Mock(return_value=CommandType.ABS_EE_POSE),
        get_quaternion_indices=Mock(return_value=(3,)),
        get_binary_action_values=Mock(return_value={}),
    )
    generator = PoseTargetMotionGenerator(embodiment)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_manager=SimpleNamespace(total_action_dim=7, active_terms=["arm"], action_term_dim=[7])
        )
    )
    goal = measured[0].numpy().copy()
    assert generator.is_goal_reached(goal, env, 0)
    measured[0] = torch.tensor(invalid_state)
    assert not generator.is_goal_reached(goal, env, 0)
