# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Categorical, build-time variation of gripper contact materials."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.contact_material import ColliderContactMaterialCfg, ContactMaterialCfg
from isaaclab_arena.utils.usd.prim_paths import parse_relative_prim_path
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@configclass
class GripperMaterialVariationCfg(VariationBaseCfg):
    """Named contact presets sampled uniformly once per scene build."""

    presets: dict[str, ContactMaterialCfg] = field(
        default_factory=lambda: {
            "low_friction": ContactMaterialCfg(static_friction=0.3, dynamic_friction=0.2),
            "medium_friction": ContactMaterialCfg(static_friction=0.7, dynamic_friction=0.5),
            "high_friction": ContactMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
        }
    )
    """Exploratory contact presets; these are not calibrated real-world material measurements."""

    material_names: list[str] = field(default_factory=list)
    """Preset names to sample; empty selects all presets, and a singleton fixes the material."""

    friction_combine_mode: str = "average"
    """Fixed PhysX combine rule for every preset; Newton uses its own contact combination."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Categorical sampler whose recorded value is the chosen preset name."""


class GripperMaterialVariation(BuildTimeVariationBase):
    """Apply one sampled material to all selected finger-pad colliders before cloning.

    Args:
        embodiment: Robot owning the gripper and its spawn addons.
        collider_paths: Exact collider paths relative to the selected robot's USD root.
        cfg: Material pool and sampling configuration; disabled by default.
        scene_key: Embodiment scene entry containing the gripper.
        name: Variation identifier on the embodiment.
    """

    cfg: GripperMaterialVariationCfg

    def __init__(
        self,
        embodiment: EmbodimentBase,
        collider_paths: tuple[str, ...],
        cfg: GripperMaterialVariationCfg | None = None,
        scene_key: str = "robot",
        name: str = "gripper_material",
    ):
        super().__init__(cfg=cfg if cfg is not None else GripperMaterialVariationCfg(), name=name)
        assert not isinstance(collider_paths, str), "Provide a sequence of collider paths, not a single string."
        paths = tuple(str(parse_relative_prim_path(path)) for path in collider_paths)
        assert paths and len(set(paths)) == len(paths), "Provide unique finger-pad colliders."
        self._embodiment = embodiment
        self._collider_paths = paths
        self._scene_key = scene_key

    def _realize_at_build_time(self) -> None:
        names = self.cfg.material_names or list(self.cfg.presets)
        assert names and len(set(names)) == len(names), "Provide a nonempty pool of unique material names."
        assert self.cfg.friction_combine_mode in {"average", "min", "multiply", "max"}, "Invalid friction combine mode."
        for name in names:
            assert name in self.cfg.presets, f"Unknown gripper material preset: {name!r}"
            self.cfg.presets[name].validate_coefficients()
        robot_cfg = getattr(self._embodiment.scene_config, self._scene_key, None)
        assert robot_cfg is not None, f"Unknown gripper scene entry: {self._scene_key!r}"
        assert isinstance(robot_cfg.spawn, UsdFileCfg), "Gripper material variation requires a USD spawn config."

        # Embodiment addons apply after backend defaults, before scene composition and spawning.
        addons = deepcopy(self._embodiment.spawn_cfg_addon.get(self._scene_key, {}))
        overrides = addons.setdefault("prim_physics", {})
        for path in self._collider_paths:
            existing = overrides.get(path, getattr(robot_cfg.spawn, "prim_physics", {}).get(path))
            assert existing is None or isinstance(
                existing, ColliderContactMaterialCfg
            ), f"Gripper material variation would replace another physics override at {path!r}."
        material_name = self.sampler.sample(num_samples=1, choices=names)[0]
        for path in self._collider_paths:
            overrides[path] = ColliderContactMaterialCfg(
                material=deepcopy(self.cfg.presets[material_name]),
                friction_combine_mode=self.cfg.friction_combine_mode,
            )
        addons["make_uninstanceable"] = True
        self._embodiment.spawn_cfg_addon[self._scene_key] = addons
