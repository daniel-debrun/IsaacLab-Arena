# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted VLM inference with configurable observation and action adapters."""

from __future__ import annotations

import base64
import importlib
import io
import json
import numpy as np
import os
import torch
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from PIL import Image

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.inference.backend import INTERNAL_ENDPOINT
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent_policy.embodiment_adapter.adapter import AgentActionAdapter, VLMObservationAdapter


def encode_image(image: np.ndarray, max_edge: int = 384) -> str:
    """Encode an RGB or RGBA image as JPEG, preserving aspect ratio without upscaling.

    Args:
        image: HWC uint8 pixels; the alpha channel, if present, is discarded.
        max_edge: Maximum encoded image width or height.

    Returns:
        A JPEG data URL for an image-capable chat-completion endpoint.
    """
    assert isinstance(max_edge, int) and max_edge > 0, "max_edge must be a positive integer"
    assert image.ndim == 3 and image.shape[-1] in (3, 4), f"Unexpected image shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 pixels, got {image.dtype}"
    frame = Image.fromarray(image[..., :3])
    frame.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _adapter_type(class_path: str, base_type: type) -> type:
    """Resolve a configured adapter class and check its interface."""
    assert class_path and "." in class_path, f"Configure a dotted class path for {base_type.__name__}"
    module_name, class_name = class_path.rsplit(".", 1)
    adapter_type = getattr(importlib.import_module(module_name), class_name)
    assert isinstance(adapter_type, type) and issubclass(
        adapter_type, base_type
    ), f"{class_path} must inherit {base_type.__name__}"
    return adapter_type


@dataclass
class VLMAgentPolicyCfg(PolicyCfg):
    """Configure hosted inference and the observation/action contracts."""

    system_prompt: str = ""
    """Required system message defining the task, units, and response contract."""

    observation_adapter: str = ""
    """Adapter class path, constructed with image_max_edge unless supplied at runtime."""

    action_adapter: str = ""
    """Adapter class path, constructed with action_adapter_kwargs unless supplied at runtime."""

    action_adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    """Constructor arguments for the configured action adapter."""

    model: str = INTERNAL_ENDPOINT.model
    """Model served by the configured chat-completion endpoint."""

    base_url: str = INTERNAL_ENDPOINT.base_url
    """OpenAI-compatible endpoint supporting images and JSON-schema responses."""

    api_key_env_var: str = INTERNAL_ENDPOINT.api_key_env_var
    """Environment variable containing the endpoint credential."""

    image_max_edge: int = 384
    """Maximum image width or height sent to the endpoint."""

    timeout_s: float = 180.0
    """Timeout for an inference request."""

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    def __post_init__(self):
        assert (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ), "Set policy.system_prompt in the experiment"
        assert isinstance(self.image_max_edge, int) and self.image_max_edge > 0
        assert self.timeout_s > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1


@register_policy
class VLMAgentPolicy(PolicyBase[VLMAgentPolicyCfg]):
    """Request one action per environment per step using stateless adapters.

    Each request contains only the current images, robot state, and task description.
    The adapters define the command dimensions, units, and native action conversion.
    """

    name = "vlm_agent"

    def __init__(
        self,
        config: VLMAgentPolicyCfg,
        *,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter | None = None,
    ):
        super().__init__(config)
        if observation_adapter is None:
            observation_adapter = _adapter_type(config.observation_adapter, VLMObservationAdapter)(
                image_max_edge=config.image_max_edge
            )
        if action_adapter is None:
            action_adapter = _adapter_type(config.action_adapter, AgentActionAdapter)(**config.action_adapter_kwargs)
        assert isinstance(observation_adapter, VLMObservationAdapter)
        assert isinstance(action_adapter, AgentActionAdapter)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        api_key = os.environ.get(config.api_key_env_var)
        assert api_key, f"Set {config.api_key_env_var} for hosted inference"
        self._client = OpenAI(base_url=config.base_url, api_key=api_key, timeout=config.timeout_s, max_retries=2)
        self.task_description = None

    @property
    def is_remote(self) -> bool:
        return True

    def get_action(self, env, observation) -> torch.Tensor:
        """Infer a batch of native actions from the current observation.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Batched observations with uint8 HWC images in camera_obs.

        Returns:
            Actions shaped (num_envs, action_dim) on the environment device.
        """
        assert self.task_description, "Set the task description before requesting actions"
        self.action_adapter.validate_environment(env)
        states = self.observation_adapter.extract_proprioception(env, observation)
        assert len(states) == env.unwrapped.num_envs, "Expected one state per environment"
        actions = []
        for env_id, state in enumerate(states):
            content = [{"type": "text", "text": json.dumps({"task": self.task_description, "proprioception": state})}]
            for key in self.observation_adapter.camera_keys:
                images = observation["camera_obs"][key]
                assert images.ndim == 4 and images.shape[0] == env.unwrapped.num_envs, "Expected batched HWC images"
                image = images[env_id].detach().cpu().numpy()
                content.extend([
                    {"type": "text", "text": f"camera={key}"},
                    {"type": "image_url", "image_url": {"url": encode_image(image, self.config.image_max_edge)}},
                ])
            command = self._infer(content, state)
            action = self.action_adapter.command_to_action(env, command)
            assert (
                action.ndim == 1 and action.numel() > 0 and torch.isfinite(action).all()
            ), "Expected a finite action vector"
            actions.append(action.to(device=env.unwrapped.device))
        return torch.stack(actions)

    def _infer(self, content: list[dict], state: dict) -> np.ndarray:
        """Request a command, allowing the model to correct locally rejected responses."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content},
        ]
        for attempt in range(self.config.max_attempts):
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_completion_tokens=8192,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_actions",
                        "strict": True,
                        "schema": self.action_adapter.response_schema,
                    },
                },
            )
            choice = response.choices[0]
            raw = choice.message.content
            try:
                assert choice.finish_reason == "stop", f"Incomplete response: {choice.finish_reason}"
                assert raw, f"No actions returned; refusal={choice.message.refusal}"
                command = np.asarray(self.action_adapter.decode_command(json.loads(raw), state), dtype=np.float32)
                assert command.ndim == 1 and command.size > 0, "Expected one command vector"
                assert np.isfinite(command).all(), "Decoded command must be finite"
                return command
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                if attempt == self.config.max_attempts - 1:
                    raise
                messages.append({"role": "assistant", "content": raw or ""})
                messages.append({"role": "user", "content": f"Invalid actions: {exc}. Return a corrected prediction."})

    def close(self) -> None:
        """Release the hosted inference client."""
        self._client.close()
