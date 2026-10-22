# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured observation selection for VLM policies."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena_vlm_agent.image_utils import extract_rgb_images_from_camera_observations, image_to_jpeg_data_url

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

POLICY_OBSERVATION_GROUP = "policy"
"""Observation group containing the embodiment's named state terms."""

CAMERA_OBSERVATION_GROUP = "camera_obs"
"""Observation group containing camera images indexed by camera key."""


class VLMObservationAdapter(ABC):
    """Convert an embodiment's observations into model input content."""

    def __init__(self, embodiment: EmbodimentBase):
        """Configure the robot supplying observations.

        Args:
            embodiment: Robot defining state and camera observations.
        """
        self.embodiment = embodiment

    @abstractmethod
    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Build modality-specific model inputs for one environment.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched state and camera observations.
            env_id: Environment index to prepare the request for.
            image_max_width_or_height: Maximum encoded image width or height, in pixels.

        Returns:
            State text, camera labels, and JPEG images for the selected modalities.
        """


class StateObservationAdapter(VLMObservationAdapter):
    """Include named robot state and command-body poses for one environment."""

    def __init__(self, embodiment: EmbodimentBase):
        super().__init__(embodiment)
        self._command_bodies: dict[str, tuple[str, str]] = {}
        observation_cfg = embodiment.get_observation_cfg()
        state_cfg = getattr(observation_cfg, POLICY_OBSERVATION_GROUP, None)
        assert state_cfg is not None, (
            "State modality requires embodiment.get_observation_cfg().policy: an Isaac Lab observation "
            "group named 'policy' containing robot state terms (e.g. joint_pos and gripper_pos). "
            "The adapter reads those terms from observation['policy'][term_name]."
        )
        assert not state_cfg.concatenate_terms, (
            "State modality requires unconcatenated policy observations: set "
            "embodiment.get_observation_cfg().policy.concatenate_terms=False so observation['policy'] "
            "is a dictionary of named state tensors rather than one concatenated tensor."
        )
        self.state_keys = tuple(name for name, term in vars(state_cfg).items() if hasattr(term, "func"))
        action_cfg = embodiment.get_action_cfg()
        if action_cfg is not None:
            for name, term in vars(action_cfg).items():
                if getattr(term, "body_name", None):
                    self._command_bodies[name] = (term.asset_name, term.body_name)

    def extract_robot_state(self, env, observation, env_id: int) -> dict:
        """Copy configured state terms and command-body poses for one environment.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Named policy observation tensors with environment as the first dimension.
            env_id: Environment index within the batched observations.

        Returns:
            Policy terms and root-relative command-body poses, indexed by their configuration names.
        """
        base = env.unwrapped
        state = {}
        for key in self.state_keys:
            batch = observation[POLICY_OBSERVATION_GROUP][key]
            assert batch.ndim >= 1 and batch.shape[0] == base.num_envs, "Expected one state entry per environment"
            state[key] = batch[env_id].detach().cpu().tolist()
        poses = {}
        for name, (asset_name, body_name) in self._command_bodies.items():
            # T_R_E maps the configured command body E into its articulation root frame R.
            T_R_E = base.arena_world.get_body_pose_in_root(asset_name, body_name)
            assert T_R_E.shape == (base.num_envs, 7), "Expected one root-relative body pose per environment"
            pose = T_R_E[env_id].detach().cpu().tolist()
            poses[name] = {
                "asset_name": asset_name,
                "body_name": body_name,
                "position_root_m": pose[:3],
                "quaternion_root_xyzw": pose[3:],
            }
        return {"policy": state, "command_body_poses": poses}

    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Return the selected environment's state as JSON text."""
        return [{
            "type": "text",
            "text": json.dumps(self.extract_robot_state(env, observation, env_id), allow_nan=False),
        }]


class ImageObservationAdapter(VLMObservationAdapter):
    """Include the configured current RGB camera views for one environment."""

    def __init__(self, embodiment: EmbodimentBase):
        super().__init__(embodiment)
        observation_cfg = embodiment.get_observation_cfg()
        camera_cfg = getattr(observation_cfg, CAMERA_OBSERVATION_GROUP, None)
        assert camera_cfg is not None, "RGB modality requires camera observations; enable the embodiment's cameras"
        self.camera_keys = tuple(
            name
            for name, term in vars(camera_cfg).items()
            if hasattr(term, "func") and term.params.get("data_type") == "rgb"
        )
        assert self.camera_keys, "RGB modality requires at least one configured RGB observation"

    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Return labeled, resized RGB views for the selected environment."""
        content = []
        images = extract_rgb_images_from_camera_observations(
            observation[CAMERA_OBSERVATION_GROUP], self.camera_keys, env_id
        )
        for key, image in images.items():
            content.extend([
                {"type": "text", "text": f"camera={key}"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_jpeg_data_url(image, image_max_width_or_height)},
                },
            ])
        return content


class ImageStateObservationAdapter(VLMObservationAdapter):
    """Include robot state followed by current RGB camera views."""

    def __init__(self, embodiment: EmbodimentBase):
        super().__init__(embodiment)
        self.state_adapter = StateObservationAdapter(embodiment)
        self.image_adapter = ImageObservationAdapter(embodiment)

    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Return state text and labeled RGB views for the selected environment."""
        return self.state_adapter.build_request_content(
            env, observation, env_id, image_max_width_or_height=image_max_width_or_height
        ) + self.image_adapter.build_request_content(
            env, observation, env_id, image_max_width_or_height=image_max_width_or_height
        )
