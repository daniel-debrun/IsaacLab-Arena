# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check DROID pose schemas, measured state, and policy execution without inference services."""

import base64
import copy
import io
import json
import math
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ValidationError

from isaaclab_arena.inference.backend import build_strict_schema
from isaaclab_arena_vlm_agent.commands import Command, CommandChunk, CommandType
from isaaclab_arena_vlm_agent.droid_action_adapter import (
    DroidActionAdapter,
    DroidChunkActionAdapter,
    DroidGoalActionAdapter,
)
from isaaclab_arena_vlm_agent.droid_motion_generator import DroidPoseMotionGenerator
from isaaclab_arena_vlm_agent.droid_observation_adapter import DroidObservationAdapter
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMChunkPolicy, VLMGoalPolicy, VLMPolicyCfg, VLMStepPolicy


def _embodiment(scene_key="robot", body_name="base_link"):
    return SimpleNamespace(
        get_scene_key=Mock(return_value=scene_key),
        get_command_body_name=Mock(return_value=body_name),
    )


def _target(x=0.4, closed=False):
    return {"values": [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, float(closed)]}


def _state(**overrides):
    return {
        "position_root_m": [0.4, 0.0, 0.3],
        "quaternion_root_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_closed_fraction": 0.0,
    } | overrides


def _world(*states, close_position=math.pi / 4):
    poses = torch.tensor([state["position_root_m"] + state["quaternion_root_xyzw"] for state in states])
    fingers = torch.tensor([state["gripper_closed_fraction"] * close_position for state in states])
    return SimpleNamespace(
        get_body_pose_in_root=Mock(return_value=poses), get_joint_position=Mock(return_value=fingers)
    )


def _inputs():
    poses = torch.tensor([[0.4, 0, 0.3, 0, 0, 0, 1], [0.5, 0, 0.3, 0, 0, 1, 0]])
    world = SimpleNamespace(
        get_body_pose_in_root=Mock(return_value=poses),
        get_joint_position=Mock(return_value=torch.tensor([0.0, 1.03]) * (math.pi / 4)),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, device="cpu", arena_world=world))
    cameras = {}
    for index, key in enumerate(DroidObservationAdapter.camera_keys):
        cameras[key] = torch.full((2, 4, 4, 3), index * 50, dtype=torch.uint8)
    obs = {"policy": {"gripper_pos": torch.tensor([[0.0], [1.03]])}, "camera_obs": cameras}
    return env, obs


@pytest.mark.parametrize(
    "field,value",
    [
        ("values", []),
        ("values", ["invalid"]),
        ("values", 0.5),
    ],
)
def test_command_response_rejects_invalid_vector(field, value):
    with pytest.raises(ValidationError):
        Command.model_validate(_target() | {field: value})


def test_droid_response_schema_and_json_round_trip():
    command = Command.model_validate(_target(closed=True))
    assert Command.model_validate_json(command.model_dump_json()) == command
    schema = build_strict_schema(Command)
    assert schema["additionalProperties"] is False
    field = schema["properties"]["values"]
    assert field["items"] == {"type": "number"}
    assert field["minItems"] == 1
    assert set(schema["properties"]) == {"values"}


def test_observation_returns_only_root_pose_and_measured_gripper():
    env, obs = _inputs()
    adapter = DroidObservationAdapter()
    states = [adapter.extract_robot_state(env, obs, env_id) for env_id in range(2)]
    assert all(type(state) is dict for state in states)
    assert json.loads(json.dumps(states, allow_nan=False)) == states
    env.unwrapped.arena_world.get_body_pose_in_root.assert_called_with("robot", "base_link")
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
    adapter = DroidActionAdapter(_embodiment(), CommandType.EE_POSE)
    response = Command.model_validate(_target(closed=closed))
    # A different current pose must not be added to an absolute command.
    command = adapter.decode_command(response, _world(_state(position_root_m=[9.0, 8.0, 7.0])), 0)
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
    generator = DroidPoseMotionGenerator()
    adapter = DroidGoalActionAdapter(_embodiment(), CommandType.EE_POSE, generator)
    world = _world(_state(), _state(**{field: value}), _state(quaternion_root_xyzw=[0.0, 0.0, 0.0, -1.0]))
    goal = adapter.decode_command(Command.model_validate(_target()), world, 0)
    assert generator.goal_reached(goal, world, 0)
    assert not generator.goal_reached(goal, world, 1)
    assert generator.goal_reached(goal, world, 2)
    world.get_joint_position.assert_called_with("robot", "finger_joint")
    command = generator.compute_command(goal, world, 1)
    np.testing.assert_array_equal(command, goal)
    command[0] += 1
    assert goal[0] == pytest.approx(0.4)


@pytest.mark.parametrize("length", [1, 3])
def test_chunk_length_is_checked_before_execution(length):
    adapter = DroidChunkActionAdapter(_embodiment(), CommandType.EE_POSE, chunk_size=2)
    command = adapter.decode_command(
        CommandChunk.model_validate({"commands": [_target()] * length}), _world(_state()), 0
    )
    assert not adapter.is_command_valid(command)


@pytest.mark.parametrize("mode", ["single", "chunk", "goal"])
def test_droid_adapters_integrate_with_policy_and_partial_resets(mode):
    if mode == "single":
        policy_type, adapter = VLMStepPolicy, DroidActionAdapter(_embodiment(), CommandType.EE_POSE)
    elif mode == "chunk":
        policy_type, adapter = VLMChunkPolicy, DroidChunkActionAdapter(_embodiment(), CommandType.EE_POSE, chunk_size=2)
    else:
        policy_type, adapter = VLMGoalPolicy, DroidGoalActionAdapter(
            _embodiment(), CommandType.EE_POSE, DroidPoseMotionGenerator()
        )
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
        **({"max_steps_per_goal": 2} if mode == "goal" else {}),
    )
    policy.set_task_description("Move the gripper")
    env, obs = _inputs()
    first = policy.get_action(env, obs)
    torch.testing.assert_close(first[:, 0], torch.tensor([0.6, 0.6]))
    assert len(requests) == 2
    content = requests[0].messages[1]["content"]
    assert json.loads(content[1]["text"])["gripper_closed_fraction"] == 0.0
    assert [content[index]["text"] for index in (2, 4, 6)] == [
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
    adapter = DroidChunkActionAdapter(_embodiment(), CommandType.EE_POSE, chunk_size=2)
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
        ("values", [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("values", [0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("values", [0.0] * 8),
        ("values", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        ("values", [0.0, 0.0, 1.0, 0.0]),
        ("values", [0.0] * 9),
        ("values", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]),
    ],
)
def test_invalid_droid_pose_is_corrected_before_any_action(mode, field, value):
    if mode == "single":
        policy_type, adapter = VLMStepPolicy, DroidActionAdapter(_embodiment(), CommandType.EE_POSE)
    elif mode == "chunk":
        policy_type, adapter = VLMChunkPolicy, DroidChunkActionAdapter(_embodiment(), CommandType.EE_POSE, chunk_size=2)
    else:
        policy_type, adapter = VLMGoalPolicy, DroidGoalActionAdapter(
            _embodiment(), CommandType.EE_POSE, DroidPoseMotionGenerator()
        )
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
        **({"max_steps_per_goal": 2} if mode == "goal" else {}),
    )
    policy.set_task_description("Move")
    actions = policy.get_action(*_inputs())
    torch.testing.assert_close(actions[:, 0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(actions).all()
    assert backend.infer.call_count == 3


@pytest.mark.parametrize("env_id", [0, 1])
def test_droid_request_content_contains_state_and_resized_camera_views(env_id):
    env, obs = _inputs()
    for index, key in enumerate(DroidObservationAdapter.camera_keys):
        obs["camera_obs"][key][env_id].fill_(40 + index * 50)
    content = DroidObservationAdapter().build_request_content(env, obs, env_id, image_max_width_or_height=2)
    state = json.loads(content[0]["text"])
    assert state["position_root_m"][0] == pytest.approx(0.4 + env_id * 0.1)
    assert state["gripper_closed_fraction"] == pytest.approx(1.03 * env_id)
    assert len(content) == 7
    for index, key in enumerate(DroidObservationAdapter.camera_keys):
        assert content[1 + index * 2] == {"type": "text", "text": f"camera={key}"}
        url = content[2 + index * 2]["image_url"]["url"]
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert image.size == (2, 2)
        assert image.getpixel((0, 0)) == (40 + index * 50,) * 3


@pytest.mark.parametrize("close_position", [math.pi / 4, 0.461])
def test_droid_motion_generator_uses_configured_gripper_close_position(close_position):
    generator = DroidPoseMotionGenerator(gripper_closed_position_rad=close_position)
    world = _world(
        _state(gripper_closed_fraction=1.03), _state(gripper_closed_fraction=0.5), close_position=close_position
    )
    goal = DroidActionAdapter(_embodiment(), CommandType.EE_POSE).decode_command(
        Command.model_validate(_target(closed=True)), world, 0
    )
    assert generator.goal_reached(goal, world, 0)
    assert not generator.goal_reached(goal, world, 1)


def test_droid_goal_completion_reads_live_world_without_prompt_state():
    env, _ = _inputs()
    world = env.unwrapped.arena_world
    adapter = DroidGoalActionAdapter(_embodiment(), CommandType.EE_POSE, DroidPoseMotionGenerator())
    adapter.is_robot_action_space_supported = Mock(return_value=True)
    observation_adapter = DroidObservationAdapter()
    observation_adapter.build_request_content = Mock(return_value=[])
    backend = SimpleNamespace(infer=Mock(return_value=_target(0.6, closed=True)))
    policy = VLMGoalPolicy(
        VLMPolicyCfg(system_prompt="Return an absolute pose goal."),
        inference_backend=backend,
        observation_adapter=observation_adapter,
        action_adapter=adapter,
        max_steps_per_goal=10,
    )
    policy.set_task_description("Move and close")
    policy.get_action(env, {})
    world.get_body_pose_in_root.return_value[0, 0] = 0.6
    policy.get_action(env, {})
    assert backend.infer.call_count == 2  # The first gripper is still open.
    world.get_joint_position.return_value[0] = math.pi / 4
    policy.get_action(env, {})
    assert backend.infer.call_count == 3  # Only environment 0 reached its goal.
    assert observation_adapter.build_request_content.call_count == 3


def test_chunk_adapter_accepts_an_embodiment_with_default_chunk_size():
    embodiment = _embodiment()
    adapter = DroidChunkActionAdapter(embodiment, CommandType.EE_POSE)
    adapter.is_robot_action_space_supported = Mock(return_value=True)
    backend = SimpleNamespace(infer=Mock(return_value={"commands": [_target(0.6)]}))
    policy = VLMChunkPolicy(
        VLMPolicyCfg(system_prompt="Return absolute pose commands."),
        inference_backend=backend,
        observation_adapter=DroidObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move")
    env, observation = _inputs()
    for _ in range(2):
        actions = policy.get_action(env, observation)
        torch.testing.assert_close(actions[:, 0], torch.tensor([0.6, 0.6]))
    assert backend.infer.call_count == 4


def test_command_type_must_be_explicit_enum():
    with pytest.raises(AssertionError, match="CommandType"):
        DroidActionAdapter(_embodiment(), "joint_position")


def test_goal_adapter_rejects_mismatching_motion_generator():
    with pytest.raises(AssertionError, match="command types must match"):
        DroidGoalActionAdapter(_embodiment(), CommandType.JOINT_POSITION, DroidPoseMotionGenerator())


def test_command_schema_is_shared_across_coordinate_spaces():
    adapter = DroidActionAdapter(_embodiment(), CommandType.JOINT_POSITION)
    schema = build_strict_schema(adapter.response_model)
    assert set(schema["properties"]) == {"values"}
    pose_adapter = DroidActionAdapter(_embodiment(), CommandType.EE_POSE)
    assert adapter.response_model is pose_adapter.response_model is Command
    # These values are valid joint positions but would fail quaternion validation.
    response = adapter.response_model(values=[0.1, -0.2, 0.3, -1.4, 0.5, 1.6, 0.7, 1.0])
    command = adapter.decode_command(response, _world(_state()), 0)
    np.testing.assert_allclose(command, [0.1, -0.2, 0.3, -1.4, 0.5, 1.6, 0.7, 1.0])
