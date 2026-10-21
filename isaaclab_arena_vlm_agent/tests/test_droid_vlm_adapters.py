# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check DROID pose schemas, measured state, and policy execution without inference services."""

import copy
import json
import math
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from isaaclab_arena.inference.backend import build_strict_schema
from isaaclab_arena_vlm_agent.embodiment_adapter.droid_action_adapter import (
    DroidChunkActionAdapter,
    DroidGoalActionAdapter,
    DroidPoseActionAdapter,
)
from isaaclab_arena_vlm_agent.embodiment_adapter.droid_observation_adapter import DroidObservationAdapter
from isaaclab_arena_vlm_agent.embodiment_adapter.droid_pose import DroidPoseChunk, DroidPoseCommand
from isaaclab_arena_vlm_agent.policy.vlm_agent_policy import VLMChunkPolicy, VLMGoalPolicy, VLMPolicyCfg, VLMStepPolicy


def _target(x=0.4, closed=False):
    return {
        "position_root_m": [x, 0.0, 0.3],
        "quaternion_root_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_closed": closed,
    }


def _state(**overrides):
    fields = _target()
    del fields["gripper_closed"]
    return fields | {"gripper_closed_fraction": 0.0} | overrides


def _inputs():
    poses = torch.tensor([[0.4, 0, 0.3, 0, 0, 0, 1], [0.5, 0, 0.3, 0, 0, 1, 0]])
    world = SimpleNamespace(get_body_pose_in_root=Mock(return_value=poses))
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, device="cpu", arena_world=world))
    cameras = {}
    for index, key in enumerate(DroidObservationAdapter.camera_keys):
        cameras[key] = torch.full((2, 4, 4, 3), index * 50, dtype=torch.uint8)
    obs = {"policy": {"gripper_pos": torch.tensor([[0.0], [1.03]])}, "camera_obs": cameras}
    return env, obs


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_root_m", [0.1, 0.2]),
        ("quaternion_root_xyzw", [0.0, 0.0, 1.0]),
        ("gripper_closed", 0.5),
    ],
)
def test_droid_response_rejects_invalid_pose_and_gripper(field, value):
    with pytest.raises(ValidationError):
        DroidPoseCommand.model_validate(_target() | {field: value})


def test_droid_response_schema_and_json_round_trip():
    command = DroidPoseCommand.model_validate(_target(closed=True))
    assert DroidPoseCommand.model_validate_json(command.model_dump_json()) == command
    schema = build_strict_schema(DroidPoseCommand)
    assert schema["additionalProperties"] is False
    for key, length in (("position_root_m", 3), ("quaternion_root_xyzw", 4)):
        field = schema["properties"][key]
        assert field["items"] == {"type": "number"}
        assert field["minItems"] == field["maxItems"] == length
    assert schema["properties"]["gripper_closed"]["type"] == "boolean"


def test_observation_returns_only_root_pose_and_measured_gripper():
    env, obs = _inputs()
    adapter = DroidObservationAdapter()
    states = adapter.extract_policy_state(env, obs)
    assert all(type(state) is dict for state in states)
    assert json.loads(json.dumps(states, allow_nan=False)) == states
    env.unwrapped.arena_world.get_body_pose_in_root.assert_called_once_with("robot", "base_link")
    assert set(states[0]) == {"position_root_m", "quaternion_root_xyzw", "gripper_closed_fraction"}
    assert states[1]["position_root_m"] == [0.5, 0.0, pytest.approx(0.3)]
    assert states[1]["quaternion_root_xyzw"] == [0.0, 0.0, 1.0, 0.0]
    assert states[1]["gripper_closed_fraction"] == pytest.approx(1.03)
    env.unwrapped.arena_world.get_body_pose_in_root.return_value.add_(1)
    obs["policy"]["gripper_pos"].zero_()
    assert states[1]["position_root_m"][0] == 0.5
    assert states[1]["gripper_closed_fraction"] == pytest.approx(1.03)


@pytest.mark.parametrize("closed", [False, True])
def test_absolute_action_preserves_xyzw_and_binary_gripper(closed):
    adapter = DroidPoseActionAdapter()
    response = DroidPoseCommand.model_validate(_target(closed=closed))
    # A different current pose must not be added to an absolute command.
    command = adapter.decode_command(response, _state(position_root_m=[9.0, 8.0, 7.0]))
    action = adapter.command_to_action(SimpleNamespace(unwrapped=SimpleNamespace(device="cpu")), command)
    assert adapter.is_command_valid(command)
    torch.testing.assert_close(action, torch.tensor([0.4, 0, 0.3, 0, 0, 0, 1, float(closed)]))
    assert action.dtype == torch.float32


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_root_m", [0.42, 0.0, 0.3]),
        ("quaternion_root_xyzw", [0.0, 0.0, math.sin(0.1), math.cos(0.1)]),
        ("gripper_closed_fraction", 0.5),
    ],
)
def test_goal_completion_requires_measured_pose_and_gripper(field, value):
    adapter = DroidGoalActionAdapter(max_steps=10)
    goal = adapter.decode_command(DroidPoseCommand.model_validate(_target()), _state())
    assert adapter.goal_reached(goal, _state())
    assert not adapter.goal_reached(goal, _state(**{field: value}))
    assert adapter.goal_reached(goal, _state(quaternion_root_xyzw=[0.0, 0.0, 0.0, -1.0]))
    command = adapter.compute_command(goal, _state(**{field: value}))
    np.testing.assert_array_equal(command, goal)
    command[0] += 1
    assert goal[0] == pytest.approx(0.4)


@pytest.mark.parametrize("length", [1, 3])
def test_chunk_length_is_checked_before_execution(length):
    adapter = DroidChunkActionAdapter(chunk_size=2)
    command = adapter.decode_command(DroidPoseChunk.model_validate({"commands": [_target()] * length}), _state())
    assert not adapter.is_command_valid(command)


@pytest.mark.parametrize("mode", ["single", "chunk", "goal"])
def test_droid_adapters_integrate_with_policy_and_partial_resets(mode):
    if mode == "single":
        policy_type, adapter = VLMStepPolicy, DroidPoseActionAdapter()
    elif mode == "chunk":
        policy_type, adapter = VLMChunkPolicy, DroidChunkActionAdapter(chunk_size=2)
    else:
        policy_type, adapter = VLMGoalPolicy, DroidGoalActionAdapter(max_steps=2)
    # Real action-manager compatibility is covered by the simulation test.
    adapter.is_robot_action_space_supported = Mock(return_value=True)
    requests = []

    def infer(request):
        requests.append(copy.deepcopy(request))
        if mode == "chunk":
            return {"commands": [_target(0.6), _target(0.7, closed=True)]}
        return _target(0.6)

    policy = policy_type(
        VLMPolicyCfg(system_prompt="Return an absolute base_link target in robot-root coordinates."),
        inference_backend=SimpleNamespace(infer=infer),
        observation_adapter=DroidObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move the gripper")
    env, obs = _inputs()
    first = policy.get_action(env, obs)
    torch.testing.assert_close(first[:, 0], torch.tensor([0.6, 0.6]))
    assert len(requests) == 2
    content = requests[0].messages[1]["content"]
    assert json.loads(content[0]["text"])["policy_state"]["gripper_closed_fraction"] == 0.0
    assert [content[index]["text"] for index in (1, 3, 5)] == [
        f"camera={key}" for key in DroidObservationAdapter.camera_keys
    ]
    policy.reset(torch.tensor([0]))
    second = policy.get_action(env, obs)
    assert len(requests) == (4 if mode == "single" else 3)
    if mode == "chunk":
        torch.testing.assert_close(second[:, 0], torch.tensor([0.6, 0.7]))
        torch.testing.assert_close(second[:, 7], torch.tensor([0.0, 1.0]))
    policy.get_action(env, obs)
    assert len(requests) == (6 if mode == "single" else 4)


def test_invalid_droid_chunk_is_corrected_before_any_action():
    adapter = DroidChunkActionAdapter(chunk_size=2)
    adapter.is_robot_action_space_supported = Mock(return_value=True)
    backend = SimpleNamespace(
        infer=Mock(
            side_effect=[
                {"commands": [_target()]},
                {"commands": [_target(0.6), _target(0.7)]},
                {"commands": [_target(0.8), _target(0.9)]},
            ]
        )
    )
    policy = VLMChunkPolicy(
        VLMPolicyCfg(system_prompt="Return two absolute pose commands."),
        inference_backend=backend,
        observation_adapter=DroidObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move")
    actions = policy.get_action(*_inputs())
    torch.testing.assert_close(actions[:, 0], torch.tensor([0.6, 0.8]))
    assert backend.infer.call_count == 3


@pytest.mark.parametrize("mode", ["single", "chunk", "goal"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("position_root_m", [float("nan"), 0.0, 0.0]),
        ("position_root_m", [0.0, float("inf"), 0.0]),
        ("quaternion_root_xyzw", [0.0, 0.0, 0.0, 0.0]),
        ("quaternion_root_xyzw", [0.0, 0.0, 0.0, 2.0]),
    ],
)
def test_invalid_droid_pose_is_corrected_before_any_action(mode, field, value):
    if mode == "single":
        policy_type, adapter = VLMStepPolicy, DroidPoseActionAdapter()
    elif mode == "chunk":
        policy_type, adapter = VLMChunkPolicy, DroidChunkActionAdapter(chunk_size=2)
    else:
        policy_type, adapter = VLMGoalPolicy, DroidGoalActionAdapter(max_steps=2)
    adapter.is_robot_action_space_supported = Mock(return_value=True)
    responses = [_target() | {field: value}, _target(0.6), _target(0.8)]
    if mode == "chunk":
        responses = [{"commands": [response, _target()]} for response in responses]
    backend = SimpleNamespace(infer=Mock(side_effect=responses))
    policy = policy_type(
        VLMPolicyCfg(system_prompt="Return absolute pose commands."),
        inference_backend=backend,
        observation_adapter=DroidObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move")
    actions = policy.get_action(*_inputs())
    torch.testing.assert_close(actions[:, 0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(actions).all()
    assert backend.infer.call_count == 3
