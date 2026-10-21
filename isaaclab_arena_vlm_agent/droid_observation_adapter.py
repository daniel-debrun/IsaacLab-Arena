# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measured DROID end-effector pose, gripper closure, and RGB camera selection."""

import json

from isaaclab_arena_vlm_agent.image_utils import extract_rgb_images_from_camera_observations, image_to_jpeg_data_url
from isaaclab_arena_vlm_agent.observation_adapter import CAMERA_OBSERVATION_GROUP, VLMObservationAdapter

POLICY_OBSERVATION_GROUP = "policy"
"""Observation group containing measured DROID robot state."""

GRIPPER_POSITION_KEY = "gripper_pos"
"""Normalized measured gripper closure within the policy observation group."""


class DroidObservationAdapter(VLMObservationAdapter):
    """Select DROID RGB views and extract only the measured pose and gripper state."""

    camera_keys = ("external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb")

    def extract_robot_state(self, env, observation, env_id: int) -> dict:
        """Return measured T_R_E and normalized gripper closure for one environment.

        Args:
            env: Arena environment with DROID registered as robot.
            observation: Unconcatenated DROID observations, including policy.gripper_pos.
            env_id: Environment index within the batched observations.

        Returns:
            A dictionary: position_root_m is the Robotiq base_link
            translation in meters relative to robot root, quaternion_root_xyzw is
            its XYZW rotation, and gripper_closed_fraction is measured finger closure
            (zero open, one closed, with physical overshoot preserved). Poses exclude
            grasp and controller offsets.
        """
        base = env.unwrapped
        T_R_E = base.arena_world.get_body_pose_in_root("robot", "base_link")
        gripper_closed_fraction = observation[POLICY_OBSERVATION_GROUP][GRIPPER_POSITION_KEY]
        assert T_R_E.shape == (base.num_envs, 7), "Expected one root-relative end-effector pose per environment"
        assert gripper_closed_fraction.shape == (
            base.num_envs,
            1,
        ), "Expected one measured gripper position per environment"
        pose = T_R_E[env_id].detach().cpu().tolist()
        return {
            "position_root_m": pose[:3],
            "quaternion_root_xyzw": pose[3:],
            "gripper_closed_fraction": gripper_closed_fraction[env_id, 0].item(),
        }

    def build_request_content(self, env, observation, env_id: int, *, image_max_width_or_height: int) -> list[dict]:
        """Combine measured DROID state and three labeled RGB views for one environment.

        Args:
            env: Arena environment with DROID registered as robot.
            observation: Batched robot state and camera observations.
            env_id: Environment index to prepare the request for.
            image_max_width_or_height: Maximum encoded image width or height, in pixels.

        Returns:
            Robot-state text followed by camera labels and JPEG image inputs.
        """
        content = [{
            "type": "text",
            "text": json.dumps(self.extract_robot_state(env, observation, env_id), allow_nan=False),
        }]
        images = extract_rgb_images_from_camera_observations(
            observation[CAMERA_OBSERVATION_GROUP], self.camera_keys, env_id
        )
        for key, image in images.items():
            content.extend([
                {"type": "text", "text": f"camera={key}"},
                {"type": "image_url", "image_url": {"url": image_to_jpeg_data_url(image, image_max_width_or_height)}},
            ])
        return content
