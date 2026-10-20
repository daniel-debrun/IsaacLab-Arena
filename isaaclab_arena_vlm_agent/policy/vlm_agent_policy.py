# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted VLM inference with configurable observation and action adapters."""

from __future__ import annotations

import json
import numpy as np
import torch
from collections import deque
from dataclasses import asdict, dataclass

from pydantic import ValidationError

from isaaclab_arena.inference.backend import InferenceBackend, InferenceRequest, build_strict_schema
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent.embodiment_adapter.action_adapter import AgentActionAdapter, InvalidAgentResponseError
from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import (
    CAMERA_OBSERVATION_GROUP,
    PolicyState,
    VLMObservationAdapter,
)
from isaaclab_arena_vlm_agent.image import image_to_jpeg_data_url


@dataclass
class VLMAgentPolicyCfg(PolicyCfg):
    """Configure policy prompts, image encoding, and action-validation attempts."""

    system_prompt: str = ""
    """Required system message defining the task, units, and response contract."""

    image_max_width_or_height: int = 384
    """Maximum image width or height sent to the endpoint, in pixels.

    For 16:9 images, 384 yields the 384x216 resolution used for all three camera views
    in Astranomy (Sections 2 and 3.4): https://research.nvidia.com/labs/srl/projects/astranomy/
    The report documents an experimental setting; it does not establish an Astra image-size limit.
    """

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    def __post_init__(self):
        assert (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ), "Set policy.system_prompt in the experiment"
        assert self.image_max_width_or_height > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1


class VLMAgentPolicy(PolicyBase[VLMAgentPolicyCfg]):
    """Execute decoded action chunks, requesting new actions when each environment's queue empties.

    Each request contains only the current images, robot state, and task description.
    Adapters return (horizon, action_dim) arrays; horizon 1 gives one inference per step.
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
        assert observation_adapter.camera_keys, "Configure at least one camera observation key"
        assert issubclass(observation_adapter.state_model, PolicyState), "Declare a concrete PolicyState model"
        assert (
            observation_adapter.state_model is action_adapter.state_model
        ), "Observation and action adapters must declare the same state_model"
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self._response_schema = build_strict_schema(action_adapter.response_model)
        self._inference_backend = inference_backend
        self.task_description = None
        self._chunks: dict[int, deque] = {}

    def get_action(self, env, observation) -> torch.Tensor:
        """Return a batch of native actions, using current observations when queues are empty.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Batched observations with uint8 HWC images in camera_obs.

        Returns:
            Actions shaped (num_envs, action_dim) on the environment device.
        """
        assert self.task_description, "Set the task description before requesting actions"
        assert self.action_adapter.is_robot_action_space_supported(
            env
        ), "Robot action space is unsupported by this adapter"
        states = self.observation_adapter.extract_policy_state(env, observation)
        assert len(states) == env.unwrapped.num_envs, "Expected one state per environment"
        assert all(
            type(state) is self.observation_adapter.state_model for state in states
        ), "Expected instances of the declared state_model"
        actions = []
        for env_id, state in enumerate(states):
            command = self._get_command(env, observation, env_id, state)
            action = self.action_adapter.command_to_action(env, command)
            assert (
                action.ndim == 1 and action.numel() > 0 and torch.isfinite(action).all()
            ), "Expected a finite action vector"
            actions.append(action.to(device=env.unwrapped.device))
        return torch.stack(actions)

    def _get_command(self, env, observation, env_id: int, state: PolicyState) -> np.ndarray:
        """Consume one action row, inferring a new chunk only when the queue is empty."""
        if not self._chunks.get(env_id):
            chunk = self._infer(self._request_content(env, observation, env_id, state), state)
            self._chunks[env_id] = deque(chunk)
        return self._chunks[env_id].popleft()

    def _request_content(self, env, observation, env_id: int, state: PolicyState) -> list[dict]:
        """Prepare current state and images for a new inference request."""
        content = [{
            "type": "text",
            "text": json.dumps({"task": self.task_description, "policy_state": state.model_dump(mode="json")}),
        }]
        for key in self.observation_adapter.camera_keys:
            images = observation[CAMERA_OBSERVATION_GROUP][key]
            assert images.ndim == 4 and images.shape[0] == env.unwrapped.num_envs, "Expected batched HWC images"
            image = images[env_id].detach().cpu().numpy()
            content.extend([
                {"type": "text", "text": f"camera={key}"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_jpeg_data_url(image, self.config.image_max_width_or_height)},
                },
            ])
        return content

    def _infer(self, content: list[dict], state: PolicyState) -> np.ndarray:
        """Request a command, allowing the model to correct locally rejected responses."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content},
        ]
        rejection: InvalidAgentResponseError | None = None
        for attempt in range(self.config.max_attempts):
            payload = self._inference_backend.infer(
                InferenceRequest(
                    messages=messages,
                    response_schema=self._response_schema,
                    schema_name="agent_actions",
                    retry_label="policy_action",
                )
            )
            try:
                response = self.action_adapter.response_model.model_validate(payload)
            except ValidationError as exc:
                rejection = InvalidAgentResponseError(str(exc), code="invalid_schema")
            else:
                try:
                    command = np.asarray(self.action_adapter.decode_command(response, state), dtype=np.float32)
                except InvalidAgentResponseError as exc:
                    rejection = exc
                else:
                    # Adapter implementation errors must propagate without asking the model to fix them.
                    if self.action_adapter.is_command_valid(command):
                        return command
                    rejection = InvalidAgentResponseError(
                        "Expected a nonempty, finite (horizon, action_dim) array with the adapter's required shape"
                    )
            messages.append({"role": "assistant", "content": json.dumps(payload)})
            messages.append({
                "role": "user",
                "content": f"Invalid actions: {json.dumps(asdict(rejection.details))}. Return a corrected prediction.",
            })
        assert rejection is not None, "Expected a rejection after exhausting response-validation attempts"
        raise rejection

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending actions for selected environments, or all environments."""
        if env_ids is None:
            self._chunks.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._chunks.pop(env_id, None)
