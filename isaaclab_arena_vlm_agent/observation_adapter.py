# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment-configured observation selection for VLM policies."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from isaaclab_arena_vlm_agent.image_utils import extract_rgb_images_from_camera_observations

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
    def extract_vlm_input_from_observation(
        self, env, observation, env_id: int
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Extract camera images and prompt values for one environment.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Current batched state and camera observations.
            env_id: Environment index to prepare the request for.

        Returns:
            HWC RGB images indexed by camera name and JSON-serializable prompt values.
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

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return the selected environment's state as prompt values."""
        return {}, self.extract_robot_state(env, observation, env_id)


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

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return configured RGB views for the selected environment."""
        images = extract_rgb_images_from_camera_observations(
            observation[CAMERA_OBSERVATION_GROUP], self.camera_keys, env_id
        )
        return images, {}


class ImageStateObservationAdapter(VLMObservationAdapter):
    """Include robot state followed by current RGB camera views."""

    def __init__(self, embodiment: EmbodimentBase):
        super().__init__(embodiment)
        self.state_adapter = StateObservationAdapter(embodiment)
        self.image_adapter = ImageObservationAdapter(embodiment)

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return state and RGB views for the selected environment."""
        _, state = self.state_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        images, _ = self.image_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        return images, state
