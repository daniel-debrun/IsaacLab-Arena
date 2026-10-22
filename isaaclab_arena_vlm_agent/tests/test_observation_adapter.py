# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify embodiment-based state and image selection without robot-specific field names."""

import json
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_vlm_agent.observation_adapter import ObservationModality, VLMObservationAdapter


def _inputs():
    state_cfg = SimpleNamespace(
        concatenate_terms=False,
        left_joints=SimpleNamespace(func=Mock()),
        right_joints=SimpleNamespace(func=Mock()),
        disabled_term=None,
    )
    camera_cfg = SimpleNamespace(
        head_color=SimpleNamespace(func=Mock(), params={"data_type": "rgb"}),
        hand_color=SimpleNamespace(func=Mock(), params={"data_type": "rgb"}),
        head_depth=SimpleNamespace(func=Mock(), params={"data_type": "distance_to_image_plane"}),
    )
    action_cfg = SimpleNamespace(
        left_target=SimpleNamespace(asset_name="left_robot", body_name="left_tool"),
        right_target=SimpleNamespace(asset_name="right_robot", body_name="right_tool"),
        joints=SimpleNamespace(asset_name="torso"),
    )
    embodiment = SimpleNamespace(
        get_observation_cfg=Mock(return_value=SimpleNamespace(policy=state_cfg, camera_obs=camera_cfg)),
        get_action_cfg=Mock(return_value=action_cfg),
    )

    def get_pose(asset_name, body_name):
        x = 1.0 if asset_name == "left_robot" else 2.0
        return torch.tensor([[x, 0, 0, 0, 0, 0, 1], [x + 10, 0, 0, 0, 0, 0, 1]])

    world = SimpleNamespace(get_body_pose_in_root=Mock(side_effect=get_pose))
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, arena_world=world))
    observation = {
        "policy": {
            "left_joints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "right_joints": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "task_only_term": torch.ones(2, 3),
        },
        "camera_obs": {
            "head_color": torch.zeros(2, 4, 8, 3, dtype=torch.uint8),
            "hand_color": torch.full((2, 8, 4, 3), 255, dtype=torch.uint8),
        },
    }
    return embodiment, env, observation


@pytest.mark.parametrize(
    "modalities",
    [
        (ObservationModality.STATE,),
        (ObservationModality.RGB,),
        (ObservationModality.STATE, ObservationModality.RGB),
    ],
)
def test_selects_only_requested_modalities_and_configured_terms(modalities):
    embodiment, env, observation = _inputs()
    if ObservationModality.STATE not in modalities:
        del observation["policy"]
        embodiment.get_observation_cfg.return_value.policy = None
    if ObservationModality.RGB not in modalities:
        del observation["camera_obs"]
        embodiment.get_observation_cfg.return_value.camera_obs = None
    adapter = VLMObservationAdapter(embodiment, modalities=modalities)
    assert adapter.embodiment is embodiment
    content = adapter.build_request_content(env, observation, 1, image_max_width_or_height=4)
    if ObservationModality.STATE in modalities:
        state = json.loads(content.pop(0)["text"])
        assert state["policy"] == {"left_joints": [3.0, 4.0], "right_joints": [7.0, 8.0]}
        assert set(state["command_body_poses"]) == {"left_target", "right_target"}
        assert state["command_body_poses"]["left_target"]["position_root_m"] == [11.0, 0.0, 0.0]
        assert state["command_body_poses"]["right_target"]["position_root_m"] == [12.0, 0.0, 0.0]
        env.unwrapped.arena_world.get_body_pose_in_root.assert_any_call("left_robot", "left_tool")
        env.unwrapped.arena_world.get_body_pose_in_root.assert_any_call("right_robot", "right_tool")
    else:
        embodiment.get_action_cfg.assert_not_called()
        env.unwrapped.arena_world.get_body_pose_in_root.assert_not_called()
    if ObservationModality.RGB in modalities:
        assert adapter.camera_keys == ("head_color", "hand_color")
        assert [item["text"] for item in content if item["type"] == "text"] == [
            "camera=head_color",
            "camera=hand_color",
        ]
        assert sum(item["type"] == "image_url" for item in content) == 2
    else:
        assert content == []


def test_joint_only_state_does_not_require_command_body_queries():
    embodiment, env, observation = _inputs()
    embodiment.get_action_cfg.return_value = SimpleNamespace(joints=SimpleNamespace(asset_name="humanoid"))
    adapter = VLMObservationAdapter(embodiment, modalities=(ObservationModality.STATE,))
    state = adapter.extract_robot_state(env, observation, 0)
    assert state["command_body_poses"] == {}
    assert state["policy"]["right_joints"] == [5.0, 6.0]
    env.unwrapped.arena_world.get_body_pose_in_root.assert_not_called()


def test_state_modality_requires_named_observation_terms():
    embodiment, _, _ = _inputs()
    embodiment.get_observation_cfg.return_value.policy.concatenate_terms = True
    with pytest.raises(AssertionError, match="unconcatenated"):
        VLMObservationAdapter(embodiment, modalities=(ObservationModality.STATE,))


def test_rgb_modality_requires_configured_rgb_views():
    embodiment, _, _ = _inputs()
    camera_cfg = embodiment.get_observation_cfg.return_value.camera_obs
    del camera_cfg.head_color
    del camera_cfg.hand_color
    with pytest.raises(AssertionError, match="configured RGB"):
        VLMObservationAdapter(embodiment, modalities=(ObservationModality.RGB,))
