# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that chooses which candidate asset a rigid Object spawns in a build."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object, assert_named_spawn_cfg
from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    random_choice: bool = False
    """False takes name number ``seed % n`` of the ``n`` names in use, so consecutive build seeds take them in
    order. True draws one name at random, seeded by the build seed and the variation key."""

    asset_names: list[str] = field(default_factory=list)
    """Candidate names to choose from, in this order. Empty uses every candidate in declaration order."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Sampler that draws the candidate name."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Choose one of several assets for a rigid Object; every environment of a build spawns the chosen asset.

    Scene export and ObjectReference read the Object's own usd_path, so they fail for an Object declared without one.

    Args:
        rigid_object: Object whose ``object_cfg.spawn`` the selection sets. It must be a rigid Object, not a
            RigidObjectSet, without another AssetSelectionVariation.
        asset_candidates: (name, spawn_cfg) pairs to choose from, for example from AssetRegistry.get_named_spawn_cfg().
        cfg: Selection settings. By default the selection is off and uses every candidate in order.
        name: Name of the variation on its Object.
    """

    cfg: AssetSelectionVariationCfg

    def __init__(
        self,
        rigid_object: Object,
        asset_candidates: list[tuple[str, SpawnerCfg]],
        cfg: AssetSelectionVariationCfg | None = None,
        name: str = "asset_selection",
    ):
        assert not isinstance(rigid_object, RigidObjectSet), (
            f"'{rigid_object.name}' is a RigidObjectSet, which already spawns one member per environment."
            " Add AssetSelectionVariation to a rigid Object."
        )
        assert isinstance(rigid_object, Object) and rigid_object.object_type == ObjectType.RIGID, (
            f"AssetSelectionVariation needs a rigid Object, but '{rigid_object.name}' ({type(rigid_object).__name__})"
            " is not one."
        )
        assert not any(
            isinstance(variation, AssetSelectionVariation) for variation in rigid_object.get_variations()
        ), f"'{rigid_object.name}' already has an AssetSelectionVariation; put all candidates in that one."
        assert asset_candidates, "AssetSelectionVariation needs at least one (name, spawn_cfg) candidate."
        # Filled before super().__init__(), whose apply_cfg call checks asset_names against these names.
        self._spawn_cfgs_by_name: dict[str, SpawnerCfg] = {}
        for candidate_index, candidate in enumerate(asset_candidates):
            assert_named_spawn_cfg(candidate, owner=f"AssetSelectionVariation '{name}' candidate {candidate_index}")
            candidate_name, spawn_cfg = candidate
            assert candidate_name not in self._spawn_cfgs_by_name, f"Duplicate candidate name '{candidate_name}'."
            self._spawn_cfgs_by_name[candidate_name] = deepcopy(spawn_cfg)
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)
        # rigid_object.variations holds this selection once it is added, and Isaac Lab's configclass validation
        # follows attributes without a cycle check, so a validated config must not hold rigid_object.
        self._rigid_object = rigid_object
        self._build_seed: int | None = None

    def apply_cfg(self, cfg: AssetSelectionVariationCfg) -> None:
        """Apply ``cfg`` after checking that ``asset_names`` holds declared candidate names, each once.

        Args:
            cfg: Selection settings to apply.
        """
        declared_names = list(self._spawn_cfgs_by_name)
        unknown_names = [asset_name for asset_name in cfg.asset_names if asset_name not in self._spawn_cfgs_by_name]
        assert not unknown_names, (
            f"AssetSelectionVariation '{self.name}' has unknown asset_names {unknown_names}; declared candidates:"
            f" {declared_names}."
        )
        assert len(set(cfg.asset_names)) == len(
            cfg.asset_names
        ), f"AssetSelectionVariation '{self.name}' repeats a name in asset_names {cfg.asset_names}."
        super().apply_cfg(cfg)

    def configure_at_build_time(self, seed: int, variation_key: str) -> None:
        # In-order selection takes name number seed % n, and VariationBase does not keep the seed.
        self._build_seed = seed
        super().configure_at_build_time(seed, variation_key)

    def _realize_at_build_time(self) -> None:
        # Checked at build time because bounding_box can be set after add_variation.
        assert self._rigid_object.bounding_box is None, (
            f"'{self._rigid_object.name}' sets a bounding_box override, which placement would use for every candidate."
            " Set bounding_box to None so placement reads the bounds of the selected asset."
        )
        names_in_use = list(self.cfg.asset_names) or list(self._spawn_cfgs_by_name)
        # Both modes call self.sampler.sample(), which reports the name to the variation recorder and returns the
        # recorded name during replay.
        if self.cfg.random_choice:
            selected_name = self.sampler.sample(1, names_in_use)[0]
        else:
            selected_name = self.sampler.sample(1, names_in_use, index=self._build_seed % len(names_in_use))[0]
        self._rigid_object.object_cfg.spawn = deepcopy(self._spawn_cfgs_by_name[selected_name])
