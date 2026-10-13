# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in pitched anchor bounding boxes for relation placement."""

from __future__ import annotations

import pytest


def test_pitched_anchor_asserts_without_opt_in():
    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    class PitchedAnchor(PlaceableAsset):
        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            return AxisAlignedBoundingBox((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))

        def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
            return self.get_bounding_box().translated((0.0, 0.0, 1.0))

    anchor = PitchedAnchor(name="hook")
    anchor.initial_pose = Pose(position_xyz=(0.0, 0.0, 1.0), rotation_xyzw=(0.130526, 0.0, 0.991445, 0.0))
    anchor.add_relation(IsAnchor())

    with pytest.raises(AssertionError, match="hook"):
        build_per_env_bounding_boxes([anchor], 1, allow_arbitrary_anchor_rotation=False)


def test_pitched_anchor_uses_world_bbox_with_opt_in():
    import torch

    from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    class PitchedAnchor(PlaceableAsset):
        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            return AxisAlignedBoundingBox((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))

        def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
            return self.get_bounding_box().translated((0.5, 0.0, 1.0))

    anchor = PitchedAnchor(name="hook")
    anchor.initial_pose = Pose(position_xyz=(0.0, 0.0, 1.0), rotation_xyzw=(0.130526, 0.0, 0.991445, 0.0))
    anchor.add_relation(IsAnchor())

    boxes = build_per_env_bounding_boxes([anchor], 2, allow_arbitrary_anchor_rotation=True)
    world = anchor.get_world_bounding_box()
    torch.testing.assert_close(boxes.object_bboxes[anchor].min_point[0], world.min_point[0])
    torch.testing.assert_close(boxes.object_bboxes[anchor].max_point[1], world.max_point.expand(2, 3)[1])
