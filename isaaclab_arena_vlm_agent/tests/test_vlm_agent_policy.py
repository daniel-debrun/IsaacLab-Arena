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
import subprocess
import sys
import textwrap
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from isaaclab_arena.inference.backend import InferenceBackend, build_strict_schema
from isaaclab_arena_vlm_agent.embodiment_adapter.action_adapter import AgentActionAdapter, InvalidAgentResponseError
from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import (
    CAMERA_OBSERVATION_GROUP,
    PolicyState,
    VLMObservationAdapter,
)
from isaaclab_arena_vlm_agent.policy.vlm_agent_policy import VLMAgentPolicy, VLMAgentPolicyCfg


class JointPositionState(PolicyState):
    joint_positions: list[float]
    """Measured joint positions in the same joint order and units as the target response."""


class ObservationAdapter(VLMObservationAdapter[JointPositionState]):
    state_model = JointPositionState
    camera_keys = ("front_view",)

    def extract_policy_state(self, env, observation) -> list[JointPositionState]:
        return [JointPositionState(joint_positions=row.tolist()) for row in observation["position"]]


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"joint_positions": [1.0], "unused_velocity": [0.0]},
        {"joint_positions": [float("nan")]},
    ],
)
def test_policy_state_rejects_missing_extra_and_nonfinite_fields(fields):
    with pytest.raises(ValidationError):
        JointPositionState.model_validate(fields)


class JointTargetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: list[float]


class ActionAdapter(AgentActionAdapter[JointPositionState]):
    state_model = JointPositionState
    response_model = JointTargetResponse

    def __init__(self, scale=1.0):
        self.scale = scale

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, policy_state: JointPositionState):
        target = np.asarray(response.target, dtype=np.float32)
        if target.shape != (len(policy_state.joint_positions),):
            raise InvalidAgentResponseError("Wrong command dimensions")
        return (target - np.asarray(policy_state.joint_positions))[None]

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
        state = json.loads(kwargs["messages"][-1]["content"][0]["text"])["policy_state"]
        return response(json.dumps({"target": (np.asarray(state["joint_positions"]) + 0.5).tolist()}))

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
            assert state["policy_state"]["joint_positions"] == observation["position"][env_id].tolist()
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
        with pytest.raises(InvalidAgentResponseError, match="Wrong command dimensions"):
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


class JointChunkAdapter(ActionAdapter):
    def __init__(self, chunk_size: int):
        super().__init__()
        self.response_model = create_model(
            "JointChunkResponse",
            __config__=ConfigDict(extra="forbid", strict=True),
            actions=(list[list[float]], Field(min_length=chunk_size, max_length=chunk_size)),
        )

    def decode_command(self, response, policy_state: JointPositionState):
        if any(len(row) != len(policy_state.joint_positions) for row in response.actions):
            raise InvalidAgentResponseError("Wrong command dimensions")
        return np.asarray(response.actions, dtype=np.float32)


class JointGoalAdapter(ActionAdapter):
    """Plan a chunk of equal joint increments toward a target from the current measured position."""

    def __init__(self, max_step: float = 0.25):
        super().__init__()
        self.max_step = max_step

    def decode_command(self, response, policy_state: JointPositionState):
        remaining = super().decode_command(response, policy_state)
        if not np.isfinite(remaining).all():
            raise InvalidAgentResponseError("Expected finite joint targets")
        num_steps = max(1, int(np.ceil(np.abs(remaining).max() / self.max_step)))
        return np.repeat(remaining / num_steps, num_steps, axis=0).astype(np.float32)


def make_multi_step_policy(backend, adapter):
    policy = VLMAgentPolicy(
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
    current_state = json.loads(calls[2]["messages"][1]["content"][0]["text"])["policy_state"]
    assert current_state["joint_positions"] == observation["position"][0].tolist()
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[8.0] * action_dim, [10.0] * action_dim])
    )
    assert len(calls) == 4
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 6


@pytest.mark.parametrize(
    "commands",
    [
        [],
        [[1, 2]],
        [1, 2, 3],
        [[1, 2], [3, float("nan")], [4, 5]],
        [[1, 2], [3], [4, 5]],
    ],
)
def test_whole_chunk_is_validated_before_execution(backend, commands):
    backend.client.chat.completions.create.return_value = response(json.dumps({"actions": commands}))
    adapter = JointChunkAdapter(chunk_size=3)
    adapter.command_to_action = Mock()
    policy = make_multi_step_policy(backend, adapter)
    with pytest.raises(InvalidAgentResponseError):
        policy.get_action(*inputs())
    adapter.command_to_action.assert_not_called()
    assert backend.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("action_dim", [2, 5])
def test_goal_adapter_plans_per_environment_horizons_from_measured_state(backend, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][1]["content"][0]["text"])["policy_state"]
        # Env 0 needs two actions; env 1 needs three.
        distance = 0.5 if len(calls) == 1 else 0.75
        return response(json.dumps({"target": (np.asarray(state["joint_positions"]) + distance).tolist()}))

    backend.client.chat.completions.create.side_effect = create
    policy = make_multi_step_policy(backend, JointGoalAdapter())
    env, observation = inputs(action_dim)
    for _ in range(2):
        torch.testing.assert_close(policy.get_action(env, observation), torch.full((2, action_dim), 0.25))
        observation["position"].add_(0.25)
    assert len(calls) == 2
    policy.get_action(env, observation)
    assert len(calls) == 3  # Only env 0 needs another prediction.
    current_state = json.loads(calls[2]["messages"][1]["content"][0]["text"])["policy_state"]
    assert current_state["joint_positions"] == observation["position"][0].tolist()
    policy.reset(torch.tensor([0]))
    policy.get_action(env, observation)
    assert len(calls) == 5  # Env 0 was reset; env 1 exhausted its original chunk.
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 7


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"wrong_field": [1, 2]}, "invalid_schema"),
        ({"target": [1]}, "invalid_command"),
        ({"target": [1, float("nan")]}, "invalid_command"),
    ],
)
def test_response_rejection_exposes_structured_details(backend, payload, code):
    backend.client.chat.completions.create.return_value = response(json.dumps(payload))
    policy = make_policy(backend, max_attempts=2)
    with pytest.raises(InvalidAgentResponseError) as caught:
        policy.get_action(*inputs())
    assert caught.value.details.code == code
    assert caught.value.details.message == str(caught.value)
    correction = backend.client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert f'"code": "{code}"' in correction
    assert backend.client.chat.completions.create.call_count == 2


def test_invalid_model_actions_are_rejected_with_python_optimization():
    # This child imports no simulation runtime; -O must not disable model-output rejection.
    script = textwrap.dedent("""\
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock
        from isaaclab_arena_vlm_agent.tests.test_vlm_agent_policy import (
            ActionAdapter, ObservationAdapter, JointPositionState,
        )
        from isaaclab_arena_vlm_agent.embodiment_adapter.action_adapter import InvalidAgentResponseError
        from isaaclab_arena_vlm_agent.policy.vlm_agent_policy import VLMAgentPolicy, VLMAgentPolicyCfg

        for command in (np.ones(2), np.empty((0, 2)), np.array([[1, np.nan]])):
            backend = SimpleNamespace(infer=Mock(return_value={"target": [1, 2]}))
            adapter = ActionAdapter()
            adapter.decode_command = Mock(return_value=command)
            policy = VLMAgentPolicy(
                VLMAgentPolicyCfg(system_prompt="Control joints", max_attempts=2),
                inference_backend=backend, observation_adapter=ObservationAdapter(), action_adapter=adapter,
            )
            try:
                policy._infer([], JointPositionState(joint_positions=[0.0, 0.0]))
            except InvalidAgentResponseError as error:
                if error.details.code != "invalid_command" or backend.infer.call_count != 2:
                    raise RuntimeError("Incorrect rejection details or retry count")
            else:
                raise RuntimeError("Invalid model actions were accepted with assertions disabled")
        print("optimized rejection checks passed")
    """)
    result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optimized rejection checks passed" in result.stdout


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


def test_incompatible_state_models_fail_before_inference(backend):
    class OtherState(PolicyState):
        velocity: list[float]

    adapter = ActionAdapter()
    adapter.state_model = OtherState
    with pytest.raises(AssertionError, match="same state_model"):
        VLMAgentPolicy(
            VLMAgentPolicyCfg(system_prompt="Control these joints."),
            inference_backend=backend,
            observation_adapter=ObservationAdapter(),
            action_adapter=adapter,
        )
    backend.client.chat.completions.create.assert_not_called()


def test_untyped_observation_state_fails_before_inference(backend):
    policy = make_policy(backend)
    policy.observation_adapter.extract_policy_state = Mock(return_value=[{"joint_positions": [1.0]}] * 2)
    with pytest.raises(AssertionError, match="declared state_model"):
        policy.get_action(*inputs())
    backend.client.chat.completions.create.assert_not_called()
