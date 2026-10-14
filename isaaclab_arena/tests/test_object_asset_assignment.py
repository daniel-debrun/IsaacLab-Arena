# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-environment spawn entries of scene objects."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

if TYPE_CHECKING:
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object


def _make_cuboid_entry(size: tuple[float, float, float]) -> CuboidCfg:
    """Return a cuboid spawn entry whose rigid_props put its rigid body at the spawn root."""
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg())


def _make_multi_asset_object(name: str, sizes: list[tuple[float, float, float]]) -> Object:
    """Return a rigid object whose PinnedMultiAssetSpawnerCfg has one cuboid spawn entry per size."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg

    spawner_cfg = PinnedMultiAssetSpawnerCfg(
        assets_cfg=[_make_cuboid_entry(size) for size in sizes],
        random_choice=False,
        activate_contact_sensors=None,
    )
    return Object(name=name, object_type=ObjectType.RIGID, spawner_cfg=spawner_cfg)


def _test_object_with_several_spawn_entries_uses_each_environment_entry(simulation_app) -> bool:
    from unittest.mock import patch

    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg
    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry

    sizes = [(0.1, 0.1, 0.2), (0.2, 0.2, 0.4), (0.3, 0.3, 0.1)]
    obj = _make_multi_asset_object("boxes", sizes)
    assert obj.has_multiple_assets
    # object_cfg.spawn is the object's own copy of spawner_cfg. It keeps the subclass and has no indices yet.
    assert isinstance(obj.object_cfg.spawn, PinnedMultiAssetSpawnerCfg)
    assert obj.object_cfg.spawn is not obj.spawner_cfg
    assert obj.object_cfg.spawn.asset_indices_by_env is None
    # The inherited func resolves against Isaac Lab's wrappers module, not the subclass's module.
    assert str(obj.object_cfg.spawn.func) == "isaaclab.sim.spawners.wrappers.wrappers:spawn_multi_asset"

    with patch("isaaclab_arena.assets.object.SpawnCfgGeometry", wraps=SpawnCfgGeometry) as spawn_cfg_geometry_class:
        # One box for all environments: the tallest entry, the 0.4 m cuboid.
        bounding_box = obj.get_bounding_box()
        torch.testing.assert_close(bounding_box.min_point, torch.tensor([[-0.1, -0.1, -0.2]]))
        torch.testing.assert_close(bounding_box.max_point, torch.tensor([[0.1, 0.1, 0.2]]))
        with pytest.raises(AssertionError, match="has no spawn entry indices"):
            obj.get_bounding_box_per_env(3)

        obj.object_cfg.spawn.asset_indices_by_env = (1, 0, 1)
        expected_sizes = torch.tensor([sizes[1], sizes[0], sizes[1]])
        for _ in range(2):
            torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, expected_sizes)
        # Every entry has its rigid body at the spawn root.
        assert obj.get_contact_sensor_prim_path() == obj.prim_path
    # Repeated queries measure each distinct spawn entry once.
    assert spawn_cfg_geometry_class.call_count == len(sizes)
    # The indices are written to object_cfg.spawn only; the declared spawner_cfg keeps None.
    assert obj.spawner_cfg.asset_indices_by_env is None

    # A new size gives entry 0 a new geometry key, so environment 1, which spawns entry 0, gets new bounds.
    obj.object_cfg.spawn.assets_cfg[0].size = (0.5, 0.5, 0.5)
    expected_sizes = torch.tensor([sizes[1], (0.5, 0.5, 0.5), sizes[1]])
    torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, expected_sizes)
    return True


def test_object_with_several_spawn_entries_uses_each_environment_entry():
    assert run_function_with_persistent_simulation_app(
        _test_object_with_several_spawn_entries_uses_each_environment_entry
    )


def _test_object_set_spawns_several_members_from_a_pinned_spawner(simulation_app) -> bool:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg

    small_box = Object(name="small_box", object_type=ObjectType.RIGID, spawner_cfg=_make_cuboid_entry((0.1, 0.1, 0.1)))
    large_box = Object(name="large_box", object_type=ObjectType.RIGID, spawner_cfg=_make_cuboid_entry((0.2, 0.2, 0.2)))

    object_set = RigidObjectSet(name="boxes", objects=[small_box, large_box])
    assert isinstance(object_set.object_cfg.spawn, PinnedMultiAssetSpawnerCfg)
    assert len(object_set.object_cfg.spawn.assets_cfg) == 2
    assert object_set.object_cfg.spawn.asset_indices_by_env is None
    assert object_set.has_multiple_assets
    return True


def test_object_set_spawns_several_members_from_a_pinned_spawner():
    assert run_function_with_persistent_simulation_app(_test_object_set_spawns_several_members_from_a_pinned_spawner)
