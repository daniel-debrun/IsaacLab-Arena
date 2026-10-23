# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that picks the registered asset a rigid object spawns."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

    from isaaclab_arena.assets.object import Object


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    random_choice: bool = False
    """False takes the names in list order, one per rebuild, starting over after the last name. True draws a name at
    random, seeded by the build seed and the variation key."""

    registry_names: list[str] = field(default_factory=list)
    """Registry names of the rigid objects to choose from, in this order. Required when enabled."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Sampler that draws the asset name."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Pick the registered asset a rigid object spawns; every environment of a build spawns the same asset.

    Args:
        rigid_object: Object whose asset the selection sets.
        cfg: Selection settings. By default the selection is off.
        name: Name of the variation on its Object.
    """

    cfg: AssetSelectionVariationCfg

    def __init__(
        self,
        rigid_object: Object,
        cfg: AssetSelectionVariationCfg | None = None,
        name: str = "asset_selection",
    ):
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)
        self._rigid_object = rigid_object
        self._variation_key: str | None = None
        self._rebuild_index: int = 0

    def configure_at_build_time(self, seed: int, variation_key: str, rebuild_index: int = 0) -> None:
        # VariationBase does not keep these: the checks name the key, and in-order selection needs the index.
        self._variation_key = variation_key
        self._rebuild_index = rebuild_index
        super().configure_at_build_time(seed, variation_key, rebuild_index)

    def _realize_at_build_time(self) -> None:
        registry_names = list(self.cfg.registry_names)
        assert registry_names, (
            f"'{self._variation_key}' is enabled but its registry_names is empty. List the registered rigid objects"
            f" to choose from, for example '{self._variation_key}.registry_names=[cracker_box,sugar_box]'."
        )
        assert len(set(registry_names)) == len(
            registry_names
        ), f"'{self._variation_key}.registry_names' repeats a name in {registry_names}; list each name once."
        # Checked at build time because bounding_box can be set after the Object is created.
        assert self._rigid_object.bounding_box is None, (
            f"'{self._rigid_object.name}' sets a bounding_box override, which placement would use for every"
            " selected asset. Set bounding_box to None so placement reads the bounds of the selected asset."
        )
        # Resolve every name, not only the selected one, so that a wrong name fails in the first build.
        spawn_cfgs_by_registry_name = {
            registry_name: _get_rigid_object_spawn_cfg(registry_name) for registry_name in registry_names
        }
        # Both modes call self.sampler.sample(), which reports the name to the variation recorder and returns the
        # recorded name during replay.
        if self.cfg.random_choice:
            selected_registry_name = self.sampler.sample(1, registry_names)[0]
        else:
            in_order_index = self._rebuild_index % len(registry_names)
            selected_registry_name = self.sampler.sample(1, registry_names, index=in_order_index)[0]
        self._rigid_object.set_asset(selected_registry_name, spawn_cfgs_by_registry_name[selected_registry_name])


def _get_rigid_object_spawn_cfg(registry_name: str) -> SpawnerCfg:
    """Return a copy of the spawn configuration of the registered rigid object ``registry_name``."""
    # Imported here because isaaclab_arena.assets.object loads pxr, which must not load before SimulationApp starts,
    # and run_execution imports this module before then.
    from isaaclab_arena.assets.object import Object

    asset_registry = AssetRegistry()
    assert asset_registry.is_registered(
        registry_name
    ), f"'{registry_name}' is not a registered asset. Use the registry name of a rigid object."
    asset_class = asset_registry.get_asset_by_name(registry_name)
    assert issubclass(
        asset_class, Object
    ), f"Asset '{registry_name}' is a {asset_class.__name__}, not an Object. Use the registry name of a rigid object."
    # Check the declared type before building the asset, which for a wrong name can open or download its USD.
    declared_object_type = getattr(asset_class, "object_type", None)
    if declared_object_type is not None:
        _assert_rigid_object_type(registry_name, declared_object_type)
    registered_asset = asset_class()
    # Classes such as ProceduralCube set their object type only in __init__.
    _assert_rigid_object_type(registry_name, registered_asset.object_type)
    return deepcopy(registered_asset.object_cfg.spawn)


def _assert_rigid_object_type(registry_name: str, object_type: ObjectType) -> None:
    """Check that the registered asset ``registry_name`` of type ``object_type`` is a rigid object."""
    assert object_type == ObjectType.RIGID, (
        f"Asset '{registry_name}' is a {object_type.value} Object, not a rigid one. Use the registry name of a rigid"
        " object."
    )
