# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that chooses which registered rigid object an Object spawns in a build."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    random_choice: bool = False
    """False takes the name at index ``seed % n`` of the ``n`` names to choose from, so consecutive build seeds
    take the names in order. True draws a name at random, seeded by the build seed and the variation key."""

    asset_names: list[str] = field(default_factory=list)
    """Registered names of the rigid objects to choose from, in this order. Required when enabled."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Sampler that draws the asset name."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Choose the registered rigid object that an Object spawns; every environment of a build spawns it.

    Args:
        rigid_object: Object whose spawn configuration the selection sets.
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
        # Set before super().__init__(), whose apply_cfg call reads it.
        self._rigid_object = rigid_object
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)
        self._build_seed: int | None = None

    @property
    def _variation_key(self) -> str:
        """Key of this selection in variation overrides, for example ``pickup.asset_selection``."""
        return f"{self._rigid_object.name}.{self.name}"

    def apply_cfg(self, cfg: AssetSelectionVariationCfg) -> None:
        """Apply ``cfg`` after checking that ``asset_names`` holds each name once.

        Args:
            cfg: Selection settings to apply.
        """
        assert len(set(cfg.asset_names)) == len(
            cfg.asset_names
        ), f"'{self._variation_key}.asset_names' repeats a name in {cfg.asset_names}; list each name once."
        super().apply_cfg(cfg)

    def configure_at_build_time(self, seed: int, variation_key: str) -> None:
        # VariationBase does not keep the seed, and in-order selection needs it.
        self._build_seed = seed
        super().configure_at_build_time(seed, variation_key)

    def _realize_at_build_time(self) -> None:
        asset_names = list(self.cfg.asset_names)
        assert asset_names, (
            f"'{self._variation_key}' is enabled but its asset_names is empty. List the registered rigid objects to"
            f" choose from, for example '{self._variation_key}.asset_names=[cracker_box,sugar_box]'."
        )
        # Checked at build time because bounding_box can be set after the Object is created.
        assert self._rigid_object.bounding_box is None, (
            f"'{self._rigid_object.name}' sets a bounding_box override, which placement would use for every"
            " selected asset. Set bounding_box to None so placement reads the bounds of the selected asset."
        )
        # Resolve every name, not only the selected one, so that a wrong name fails in the first build.
        asset_registry = AssetRegistry()
        spawn_cfgs_by_name = {name: asset_registry.get_rigid_object_spawn_cfg(name) for name in asset_names}
        # Both modes call self.sampler.sample(), which reports the name to the variation recorder and returns the
        # recorded name during replay.
        if self.cfg.random_choice:
            selected_name = self.sampler.sample(1, asset_names)[0]
        else:
            in_order_index = self._build_seed % len(asset_names)
            selected_name = self.sampler.sample(1, asset_names, index=in_order_index)[0]
        self._rigid_object.object_cfg.spawn = spawn_cfgs_by_name[selected_name]
