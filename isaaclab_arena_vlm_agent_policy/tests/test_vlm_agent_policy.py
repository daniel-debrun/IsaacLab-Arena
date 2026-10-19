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

from isaaclab_arena_vlm_agent_policy.embodiment_adapter.adapter import AgentActionAdapter, VLMObservationAdapter
from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import VLMAgentPolicy, VLMAgentPolicyCfg, encode_image


class ObservationAdapter(VLMObservationAdapter):
    camera_keys = ("front_view",)

    def extract_proprioception(self, env, observation):
        return [{"position": row.tolist()} for row in observation["position"]]


class ActionAdapter(AgentActionAdapter):
    response_schema = {
        "type": "object",
        "properties": {"target": {"type": "array", "items": {"type": "number"}}},
        "required": ["target"],
        "additionalProperties": False,
    }

    def __init__(self, scale=1.0):
        self.scale = scale

    def validate_environment(self, env):
        assert env.unwrapped.action_dim > 0, "Expected controllable joints"

    def decode_command(self, payload, proprioception):
        target = np.asarray(payload["target"], dtype=np.float32)
        assert target.shape == (len(proprioception["position"]),), "Wrong command dimensions"
        return target - np.asarray(proprioception["position"])

    def command_to_action(self, env, command):
        return super().command_to_action(env, command * self.scale)


def response(raw, finish_reason="stop"):
    message = SimpleNamespace(content=raw, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


@pytest.fixture
def client(monkeypatch):
    import isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy as agent_module

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())), close=Mock())
    monkeypatch.setenv("VLM_TEST_KEY", "test-only")
    monkeypatch.setattr(agent_module, "OpenAI", Mock(return_value=client))
    return client


def make_policy(**kwargs):
    config = VLMAgentPolicyCfg(
        system_prompt="Return the joint targets using the supplied schema.",
        observation_adapter=f"{__name__}.ObservationAdapter",
        action_adapter=f"{__name__}.ActionAdapter",
        api_key_env_var="VLM_TEST_KEY",
        image_max_edge=8,
        **kwargs,
    )
    policy = VLMAgentPolicy(config)
    policy.set_task_description("Move to the target")
    return policy


def inputs(action_dim=2):
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, device="cpu", action_dim=action_dim))
    positions = torch.arange(2 * action_dim, dtype=torch.float32).reshape(2, action_dim)
    images = torch.zeros((2, 8, 16, 3), dtype=torch.uint8)
    images[0, ..., 0] = 255
    images[1, ..., 2] = 255
    return env, {"position": positions, "camera_obs": {"front_view": images}}


@pytest.mark.parametrize("selection", ["config", "runtime"])
@pytest.mark.parametrize("action_dim", [2, 5])
def test_policy_uses_adapter_actions_for_each_environment(client, selection, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][-1]["content"][0]["text"])["proprioception"]
        return response(json.dumps({"target": (np.asarray(state["position"]) + 0.5).tolist()}))

    client.chat.completions.create.side_effect = create
    config = VLMAgentPolicyCfg(
        system_prompt="Use this robot's joint units.",
        observation_adapter=f"{__name__}.ObservationAdapter" if selection == "config" else "",
        action_adapter=f"{__name__}.ActionAdapter" if selection == "config" else "",
        action_adapter_kwargs={"scale": 2},
        api_key_env_var="VLM_TEST_KEY",
        image_max_edge=8,
    )
    adapters = (
        {}
        if selection == "config"
        else {
            "observation_adapter": ObservationAdapter(image_max_edge=8),
            "action_adapter": ActionAdapter(scale=2),
        }
    )
    policy = VLMAgentPolicy(config, **adapters)
    env, observation = inputs(action_dim)
    try:
        policy.set_task_description("Move each joint")
        torch.testing.assert_close(policy.get_action(env, observation), torch.ones((2, action_dim)))
        assert policy.is_remote
        assert policy.observation_adapter.image_max_edge == 8
        assert len(calls) == 2
        for env_id, call in enumerate(calls):
            assert call["messages"][0] == {"role": "system", "content": config.system_prompt}
            assert call["response_format"]["json_schema"]["schema"] == ActionAdapter.response_schema
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
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "bad_raw,finish_reason",
    [
        ("not JSON", "stop"),
        ('{"target": [1]}', "stop"),
        ('{"target": [1, NaN]}', "stop"),
        ('{"target": [1, 2]}', "length"),
        (None, "stop"),
    ],
)
def test_invalid_responses_are_corrected_before_actions(client, bad_raw, finish_reason):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            return response(bad_raw, finish_reason)
        return response('{"target": [1, 2]}')

    client.chat.completions.create.side_effect = create
    policy = make_policy(max_attempts=2)
    env, observation = inputs()
    try:
        action = policy.get_action(env, observation)
        torch.testing.assert_close(action, torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
        assert len(calls) == 3
        assert "Invalid actions" in calls[1]["messages"][-1]["content"]
        assert len(calls[2]["messages"]) == 2  # Correction from env 0 must not leak into env 1.
    finally:
        policy.close()


def test_invalid_commands_fail_after_bounded_attempts(client):
    client.chat.completions.create.return_value = response('{"target": [1]}')
    policy = make_policy(max_attempts=2)
    policy.action_adapter.command_to_action = Mock()
    try:
        with pytest.raises(AssertionError, match="Wrong command dimensions"):
            policy.get_action(*inputs())
        assert client.chat.completions.create.call_count == 2
        policy.action_adapter.command_to_action.assert_not_called()
    finally:
        policy.close()


def test_environment_validation_precedes_inference(client):
    policy = make_policy()
    env, observation = inputs()
    env.unwrapped.action_dim = 0
    try:
        with pytest.raises(AssertionError, match="controllable joints"):
            policy.get_action(env, observation)
        client.chat.completions.create.assert_not_called()
    finally:
        policy.close()


@pytest.mark.parametrize("class_path", ["", "builtins.dict"])
def test_missing_or_invalid_adapter_fails_before_inference(class_path):
    config = VLMAgentPolicyCfg(system_prompt="Test", observation_adapter=class_path)
    with pytest.raises(AssertionError, match="VLMObservationAdapter"):
        VLMAgentPolicy(config)


@pytest.mark.parametrize("prompt", ["", " \n\t", None])
def test_missing_prompt_is_rejected(prompt):
    with pytest.raises(AssertionError, match="policy.system_prompt"):
        VLMAgentPolicyCfg(system_prompt=prompt)


@pytest.mark.parametrize("max_edge,expected_size", [(4, (4, 2)), (32, (16, 8))])
def test_image_encoding_preserves_spatial_color_and_aspect_ratio(max_edge, expected_size):
    pixels = np.zeros((8, 16, 4), dtype=np.uint8)
    pixels[:, :8, 0] = 255
    pixels[:, 8:, 2] = 255
    pixels[..., 3] = 255
    url = encode_image(pixels, max_edge)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == expected_size
    assert image.mode == "RGB"
    left, right = image.getpixel((0, 0)), image.getpixel((image.width - 1, 0))
    assert left[0] > left[2] + 100
    assert right[2] > right[0] + 100


def test_policy_config_is_registered():
    from isaaclab_arena.assets.registries import PolicyRegistry

    registry = PolicyRegistry()
    config = VLMAgentPolicyCfg(system_prompt="Test")
    assert registry.get_policy("vlm_agent") is VLMAgentPolicy
    assert registry.get_policy_type_for_cfg(config) is VLMAgentPolicy
