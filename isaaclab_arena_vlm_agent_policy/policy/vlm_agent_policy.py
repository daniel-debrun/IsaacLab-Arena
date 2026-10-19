# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted VLM inference with configurable observation and action adapters."""

from __future__ import annotations

import base64
import io
import json
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass

from PIL import Image

from isaaclab_arena.inference.backend import InferenceBackend, InferenceRequest
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent_policy.embodiment_adapter.adapter import (
    AgentActionAdapter,
    ChunkActionAdapter,
    GoalActionAdapter,
    VLMObservationAdapter,
)


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


@dataclass
class VLMAgentPolicyCfg(PolicyCfg):
    """Configure policy prompts, image encoding, and action-validation attempts."""

    system_prompt: str = ""
    """Required system message defining the task, units, and response contract."""

    image_max_edge: int = 384
    """Maximum image width or height sent to the endpoint."""

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    def __post_init__(self):
        assert (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ), "Set policy.system_prompt in the experiment"
        assert isinstance(self.image_max_edge, int) and self.image_max_edge > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1


class VLMAgentPolicy(PolicyBase[VLMAgentPolicyCfg]):
    """Request one action per environment per step using stateless adapters.

    Each request contains only the current images, robot state, and task description.
    The adapters define the command dimensions, units, and native action conversion.
    """

    def __init__(
        self,
        config: VLMAgentPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: AgentActionAdapter,
    ):
        """Create a policy with a shared inference backend and stateless adapters.

        Args:
            config: Prompt, image encoding, and action-validation settings.
            inference_backend: Caller-owned backend shared with other inference steps.
                The caller is responsible for closing its client.
            observation_adapter: Adapter selecting camera images and preparing robot state.
            action_adapter: Adapter validating responses and converting commands to native actions.
        """
        super().__init__(config)
        assert isinstance(observation_adapter, VLMObservationAdapter)
        assert isinstance(action_adapter, AgentActionAdapter)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self._inference_backend = inference_backend
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
            command = self._get_command(env, observation, env_id, state)
            action = self.action_adapter.command_to_action(env, command)
            assert (
                action.ndim == 1 and action.numel() > 0 and torch.isfinite(action).all()
            ), "Expected a finite action vector"
            actions.append(action.to(device=env.unwrapped.device))
        return torch.stack(actions)

    def _get_command(self, env, observation, env_id: int, state: dict) -> np.ndarray:
        """Infer the command to execute for one environment on this step."""
        return self._infer(self._request_content(env, observation, env_id, state), state)

    def _request_content(self, env, observation, env_id: int, state: dict) -> list[dict]:
        """Prepare current state and images for a new inference request."""
        content = [{"type": "text", "text": json.dumps({"task": self.task_description, "proprioception": state})}]
        for key in self.observation_adapter.camera_keys:
            images = observation["camera_obs"][key]
            assert images.ndim == 4 and images.shape[0] == env.unwrapped.num_envs, "Expected batched HWC images"
            image = images[env_id].detach().cpu().numpy()
            content.extend([
                {"type": "text", "text": f"camera={key}"},
                {"type": "image_url", "image_url": {"url": encode_image(image, self.config.image_max_edge)}},
            ])
        return content

    def _infer(self, content: list[dict], state: dict) -> np.ndarray:
        """Request a command, allowing the model to correct locally rejected responses."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content},
        ]
        for attempt in range(self.config.max_attempts):
            payload = self._inference_backend.infer(
                InferenceRequest(
                    messages=messages,
                    response_schema=self.action_adapter.response_schema,
                    schema_name="agent_actions",
                    retry_label="policy_action",
                )
            )
            try:
                command = np.asarray(self.action_adapter.decode_command(payload, state), dtype=np.float32)
                self.action_adapter.validate_command(command)
                return command
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                if attempt == self.config.max_attempts - 1:
                    raise
                messages.append({"role": "assistant", "content": json.dumps(payload)})
                messages.append({"role": "user", "content": f"Invalid actions: {exc}. Return a corrected prediction."})


class AgentCommandedChunkPolicy(VLMAgentPolicy):
    """Execute a validated command chunk before requesting another from current observations."""

    def __init__(
        self,
        config: VLMAgentPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: ChunkActionAdapter,
    ):
        assert isinstance(action_adapter, ChunkActionAdapter)
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )
        self._chunks: dict[int, deque] = {}

    def _get_command(self, env, observation, env_id: int, state: dict) -> np.ndarray:
        if not self._chunks.get(env_id):
            chunk = self._infer(self._request_content(env, observation, env_id, state), state)
            self._chunks[env_id] = deque(chunk)
        return self._chunks[env_id].popleft()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending commands for selected environments, or all environments."""
        if env_ids is None:
            self._chunks.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._chunks.pop(env_id, None)


class AgentCommandedGoalPolicy(VLMAgentPolicy):
    """Use adapter-defined feedback to execute a goal until completion or its step limit."""

    def __init__(
        self,
        config: VLMAgentPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: GoalActionAdapter,
    ):
        assert isinstance(action_adapter, GoalActionAdapter)
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )
        self._goals: dict[int, tuple[np.ndarray, int]] = {}

    def _get_command(self, env, observation, env_id: int, state: dict) -> np.ndarray:
        pending = self._goals.get(env_id)
        if (
            pending is None
            or pending[1] >= self.action_adapter.max_steps
            or self.action_adapter.goal_reached(pending[0], state)
        ):
            goal = self._infer(self._request_content(env, observation, env_id, state), state)
            executed = 0
        else:
            goal, executed = pending
        command = np.asarray(self.action_adapter.compute_command(goal, state), dtype=np.float32)
        assert command.ndim == 1 and command.size > 0 and np.isfinite(command).all(), "Expected a finite goal command"
        self._goals[env_id] = (goal, executed + 1)
        return command

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending goals for selected environments, or all environments."""
        if env_ids is None:
            self._goals.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._goals.pop(env_id, None)
