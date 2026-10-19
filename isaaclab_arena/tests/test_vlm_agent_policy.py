# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise hosted inference with small robot-independent adapters and a mocked client."""

import base64
import copy
import io
import json
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError

from isaaclab_arena.inference.backend import InferenceBackend, build_strict_schema
from isaaclab_arena_vlm_agent.embodiment_adapter.action_adapter import (
    AgentActionAdapter,
    ChunkActionAdapter,
    GoalActionAdapter,
    InvalidAgentResponse,
)
from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import (
    CAMERA_OBSERVATION_GROUP,
    VLMObservationAdapter,
)
from isaaclab_arena_vlm_agent.policy.vlm_agent_policy import (
    AgentCommandedChunkPolicy,
    AgentCommandedGoalPolicy,
    VLMAgentPolicy,
    VLMAgentPolicyCfg,
)


class ObservationAdapter(VLMObservationAdapter):
    camera_keys = ("front_view",)

    def extract_proprioception(self, env, observation):
        return [{"position": row.tolist()} for row in observation["position"]]


class JointTargetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: list[float]


class ActionAdapter(AgentActionAdapter):
    response_model = JointTargetResponse

    def __init__(self, scale=1.0):
        self.scale = scale

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, proprioception):
        target = np.asarray(response.target, dtype=np.float32)
        if target.shape != (len(proprioception["position"]),):
            raise InvalidAgentResponse("Wrong command dimensions")
        return target - np.asarray(proprioception["position"])

    def command_to_action(self, env, command):
        return super().command_to_action(env, command * self.scale)


def response(raw):
    message = SimpleNamespace(content=raw, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


@pytest.fixture
def backend(monkeypatch):
    import isaaclab_arena.inference.backend as backend_module

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())), close=Mock())
    client.chat.completions.create.return_value = response("OK")
    monkeypatch.setattr(backend_module, "OpenAI", Mock(return_value=client))
    backend = InferenceBackend(api_key="test-only", endpoint="internal", model="test-model", max_retries=1)
    client.chat.completions.create.reset_mock()
    yield backend
    client.close()


def make_policy(backend, **kwargs):
    config = VLMAgentPolicyCfg(
        system_prompt="Return the joint targets using the supplied schema.",
        image_max_width_or_height=8,
        **kwargs,
    )
    policy = VLMAgentPolicy(
        config,
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=ActionAdapter(),
    )
    policy.set_task_description("Move to the target")
    return policy


def inputs(action_dim=2):
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, device="cpu", action_dim=action_dim))
    positions = torch.arange(2 * action_dim, dtype=torch.float32).reshape(2, action_dim)
    images = torch.zeros((2, 8, 16, 3), dtype=torch.uint8)
    images[0, ..., 0] = 255
    images[1, ..., 2] = 255
    return env, {"position": positions, CAMERA_OBSERVATION_GROUP: {"front_view": images}}


@pytest.mark.parametrize("action_dim", [2, 5])
def test_policy_uses_adapter_actions_for_each_environment(backend, action_dim):
    client = backend.client
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][-1]["content"][0]["text"])["proprioception"]
        return response(json.dumps({"target": (np.asarray(state["position"]) + 0.5).tolist()}))

    client.chat.completions.create.side_effect = create
    config = VLMAgentPolicyCfg(
        system_prompt="Use this robot's joint units.",
        image_max_width_or_height=8,
    )
    policy = VLMAgentPolicy(
        config,
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=ActionAdapter(scale=2),
    )
    env, observation = inputs(action_dim)
    try:
        policy.set_task_description("Move each joint")
        torch.testing.assert_close(policy.get_action(env, observation), torch.ones((2, action_dim)))
        assert not policy.is_remote
        assert len(calls) == 2
        for env_id, call in enumerate(calls):
            assert call["messages"][0] == {"role": "system", "content": config.system_prompt}
            assert call["response_format"]["json_schema"]["schema"] == build_strict_schema(ActionAdapter.response_model)
            content = call["messages"][1]["content"]
            state = json.loads(content[0]["text"])
            assert state["task"] == "Move each joint"
            assert state["proprioception"]["position"] == observation["position"][env_id].tolist()
            assert content[1]["text"] == "camera=front_view"
            image = Image.open(io.BytesIO(base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1])))
            assert image.size == (8, 4)
            assert image.getpixel((0, 0))[env_id * 2] > 240
        observation["position"].add_(1)
        policy.reset(torch.tensor([0]))
        policy.get_action(env, observation)
        assert len(calls) == 4
        assert all(len(call["messages"]) == 2 for call in calls)
    finally:
        policy.close()
    client.close.assert_not_called()  # The backend belongs to the caller, not the policy.


@pytest.mark.parametrize(
    "bad_raw,needs_correction",
    [
        ("not JSON", False),
        ('{"target": [1]}', True),
        ('{"target": [1, NaN]}', True),
        (None, False),
        ('{"wrong_field": [1, 2]}', True),
        ('{"target": "not a list"}', True),
        ('{"target": [1, 2], "extra": 1}', True),
    ],
)
def test_invalid_responses_are_corrected_before_actions(backend, bad_raw, needs_correction):
    client = backend.client
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            return response(bad_raw)
        return response('{"target": [1, 2]}')

    client.chat.completions.create.side_effect = create
    policy = make_policy(backend, max_attempts=2)
    env, observation = inputs()
    try:
        action = policy.get_action(env, observation)
        torch.testing.assert_close(action, torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
        assert len(calls) == 3
        if needs_correction:
            assert "Invalid actions" in calls[1]["messages"][-1]["content"]
        else:
            assert calls[1]["messages"] == calls[0]["messages"]  # Backend retries preserve image inputs.
        assert len(calls[2]["messages"]) == 2  # Correction from env 0 must not leak into env 1.
    finally:
        policy.close()


def test_invalid_commands_fail_after_bounded_attempts(backend):
    client = backend.client
    client.chat.completions.create.return_value = response('{"target": [1]}')
    policy = make_policy(backend, max_attempts=2)
    policy.action_adapter.command_to_action = Mock()
    try:
        with pytest.raises(InvalidAgentResponse, match="Wrong command dimensions"):
            policy.get_action(*inputs())
        assert client.chat.completions.create.call_count == 2
        policy.action_adapter.command_to_action.assert_not_called()
    finally:
        policy.close()


def test_environment_validation_precedes_inference(backend):
    client = backend.client
    policy = make_policy(backend)
    env, observation = inputs()
    env.unwrapped.action_dim = 0
    try:
        with pytest.raises(AssertionError, match="Robot action space"):
            policy.get_action(env, observation)
        client.chat.completions.create.assert_not_called()
    finally:
        policy.close()


@pytest.mark.parametrize("prompt", ["", " \n\t", None])
def test_missing_prompt_is_rejected(prompt):
    with pytest.raises(AssertionError, match="policy.system_prompt"):
        VLMAgentPolicyCfg(system_prompt=prompt)


class JointChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actions: list[list[float]]


class JointChunkAdapter(ChunkActionAdapter):
    response_model = JointChunkResponse

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, proprioception):
        return np.asarray(response.actions, dtype=np.float32)


class JointGoalAdapter(GoalActionAdapter):
    response_model = JointTargetResponse

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, proprioception):
        return np.asarray(response.target, dtype=np.float32)

    def goal_reached(self, goal, proprioception):
        return np.allclose(goal, proprioception["position"])

    def compute_command(self, goal, proprioception):
        return np.clip(goal - np.asarray(proprioception["position"]), -0.25, 0.25)


def make_multi_step_policy(backend, adapter):
    policy_type = AgentCommandedChunkPolicy if isinstance(adapter, ChunkActionAdapter) else AgentCommandedGoalPolicy
    policy = policy_type(
        VLMAgentPolicyCfg(system_prompt="Control these joints.", max_attempts=1),
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move")
    policy.reset()  # Reset before the first observation must be harmless.
    return policy


@pytest.mark.parametrize("action_dim", [2, 5])
def test_chunks_replay_in_order_and_reset_independently(backend, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        start = 3 * (len(calls) - 1) + 1
        return response(json.dumps({"actions": [[value] * action_dim for value in range(start, start + 3)]}))

    backend.client.chat.completions.create.side_effect = create
    policy = make_multi_step_policy(backend, JointChunkAdapter(chunk_size=3))
    env, observation = inputs(action_dim)
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[1.0] * action_dim, [4.0] * action_dim])
    )
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[2.0] * action_dim, [5.0] * action_dim])
    )
    assert len(calls) == 2
    policy.reset(torch.tensor([0]))
    observation["position"].add_(10)
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[7.0] * action_dim, [6.0] * action_dim])
    )
    assert len(calls) == 3
    current_state = json.loads(calls[2]["messages"][1]["content"][0]["text"])["proprioception"]
    assert current_state["position"] == observation["position"][0].tolist()
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[8.0] * action_dim, [10.0] * action_dim])
    )
    assert len(calls) == 4
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 6


@pytest.mark.parametrize("commands", [[[1, 2]], [1, 2, 3], [[1, 2], [3, float("nan")], [4, 5]]])
def test_whole_chunk_is_validated_before_execution(backend, commands):
    backend.client.chat.completions.create.return_value = response(json.dumps({"actions": commands}))
    adapter = JointChunkAdapter(chunk_size=3)
    adapter.command_to_action = Mock()
    policy = make_multi_step_policy(backend, adapter)
    with pytest.raises((AssertionError, ValidationError)):
        policy.get_action(*inputs())
    adapter.command_to_action.assert_not_called()
    assert backend.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("action_dim", [2, 5])
def test_goals_use_measured_feedback_step_limit_and_partial_reset(backend, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][1]["content"][0]["text"])["proprioception"]
        return response(json.dumps({"target": (np.asarray(state["position"]) + 1).tolist()}))

    backend.client.chat.completions.create.side_effect = create
    policy = make_multi_step_policy(backend, JointGoalAdapter(max_steps=2))
    env, observation = inputs(action_dim)
    torch.testing.assert_close(policy.get_action(env, observation), torch.full((2, action_dim), 0.25))
    assert len(calls) == 2
    # Env 0 reaches its goal; env 1 is close enough to require a smaller command.
    observation["position"][0].add_(1)
    observation["position"][1].add_(0.9)
    expected = torch.stack((torch.full((action_dim,), 0.25), torch.full((action_dim,), 0.1)))
    torch.testing.assert_close(policy.get_action(env, observation), expected)
    assert len(calls) == 3
    # Env 1 exhausts its budget while env 0 continues its second goal.
    torch.testing.assert_close(policy.get_action(env, observation), torch.full((2, action_dim), 0.25))
    assert len(calls) == 4
    policy.reset(torch.tensor([0]))
    policy.get_action(env, observation)
    assert len(calls) == 5
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 7


def test_goal_controller_output_is_validated_before_conversion(backend):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    adapter = JointGoalAdapter(max_steps=2)
    adapter.compute_command = Mock(return_value=np.array([np.inf, 0]))
    adapter.command_to_action = Mock()
    policy = make_multi_step_policy(backend, adapter)
    with pytest.raises(AssertionError, match="finite goal command"):
        policy.get_action(*inputs())
    adapter.command_to_action.assert_not_called()


@pytest.mark.parametrize("error_type", [KeyError, TypeError, AssertionError, ValueError])
@pytest.mark.parametrize("method", ["decode_command", "is_command_valid"])
def test_adapter_implementation_errors_are_not_retried(backend, error_type, method):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    policy = make_policy(backend, max_attempts=3)
    error = error_type("adapter implementation bug")
    setattr(policy.action_adapter, method, Mock(side_effect=error))
    with pytest.raises(error_type) as caught:
        policy.get_action(*inputs())
    assert caught.value is error
    assert backend.client.chat.completions.create.call_count == 1


def test_empty_choices_retry_without_correction_message(backend):
    backend.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[]),
        response('{"target": [1, 2]}'),
        response('{"target": [1, 2]}'),
    ]
    policy = make_policy(backend)
    policy.get_action(*inputs())
    calls = backend.client.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["messages"] == calls[1].kwargs["messages"]


def test_missing_camera_keys_fail_at_construction(backend):
    adapter = ObservationAdapter()
    adapter.camera_keys = ()
    with pytest.raises(AssertionError, match="camera observation key"):
        VLMAgentPolicy(
            VLMAgentPolicyCfg(system_prompt="Control these joints."),
            inference_backend=backend,
            observation_adapter=adapter,
            action_adapter=ActionAdapter(),
        )
    backend.client.chat.completions.create.assert_not_called()
