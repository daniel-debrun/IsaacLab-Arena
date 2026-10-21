# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measured DROID end-effector pose, gripper closure, and RGB camera selection."""

from isaaclab_arena_vlm_agent.embodiment_adapter.observation_adapter import VLMObservationAdapter


class DroidObservationAdapter(VLMObservationAdapter):
    """Select DROID RGB views and extract only the measured pose and gripper state."""

    camera_keys = ("external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb")

    def extract_policy_state(self, env, observation) -> list[dict]:
        """Return measured T_R_E and normalized gripper closure for each environment.

        Args:
            env: Arena environment with DROID registered as robot.
            observation: Unconcatenated DROID observations, including policy.gripper_pos.

        Returns:
            One dictionary per environment: position_root_m is the Robotiq base_link
            translation in meters relative to robot root, quaternion_root_xyzw is
            its XYZW rotation, and gripper_closed_fraction is measured finger closure
            (zero open, one closed, with physical overshoot preserved). Poses exclude
            grasp and controller offsets.
        """
        base = env.unwrapped
        T_R_E = base.arena_world.get_body_pose_in_root("robot", "base_link")
        gripper_closed_fraction = observation["policy"]["gripper_pos"]
        assert T_R_E.shape == (base.num_envs, 7), "Expected one root-relative end-effector pose per environment"
        assert gripper_closed_fraction.shape == (
            base.num_envs,
            1,
        ), "Expected one measured gripper position per environment"
        poses = T_R_E.detach().cpu().tolist()
        grippers = gripper_closed_fraction.detach().cpu().tolist()
        states = []
        for pose, gripper in zip(poses, grippers, strict=True):
            states.append({
                "position_root_m": pose[:3],
                "quaternion_root_xyzw": pose[3:],
                "gripper_closed_fraction": gripper[0],
            })
        return states
