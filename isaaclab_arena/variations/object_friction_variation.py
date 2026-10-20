# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Continuous object friction sampled once before spawning."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
    UsdPhysicsRigidBodyMaterialCfg,
)
from isaaclab.utils.configclass import configclass
from pxr import Usd, UsdPhysics, UsdShade

from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg
from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_prim_physics
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


@configclass
class ObjectFrictionVariationCfg(VariationBaseCfg):
    """Configuration for build-time object friction."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[0.2], high=[1.2]))
    """Distribution of the coefficient used for both static and dynamic friction."""


class ObjectFrictionVariation(BuildTimeVariationBase):
    """Set the object's spawn material friction once for all cloned environments.

    Args:
        object: Spawned object whose material is varied.
        cfg: Sampling configuration; disabled by default.
        name: Variation identifier on the object.
    """

    cfg: ObjectFrictionVariationCfg

    def __init__(self, object: Object, cfg: ObjectFrictionVariationCfg | None = None, name: str = "friction"):
        super().__init__(cfg=cfg if cfg is not None else ObjectFrictionVariationCfg(), name=name)
        self._object = object

    def _realize_at_build_time(self) -> None:
        spawn = deepcopy(self._object.object_cfg.spawn)
        assert hasattr(spawn, "physics_material"), "Object friction requires a spawn material slot."
        assert self.sampler.shape_per_sample == (1,), "Object friction requires one scalar per sample."
        friction = float(self.sampler.sample(num_samples=1)[0, 0])
        assert math.isfinite(friction) and friction >= 0, "Friction must be finite and nonnegative."

        if isinstance(spawn, UsdFileCfg):
            # Material overrides must edit collider children, which are read-only inside USD instances.
            # WARNING: disabling USD instancing increases stage memory use by making the geometry editable.
            spawn.make_uninstanceable = True
            overrides = getattr(spawn, "prim_physics", {}).copy()
            previous = overrides.get(".")
            if isinstance(previous, _ObjectFrictionCfg):
                previous = previous.previous
                if previous is None:
                    overrides.pop(".")
                else:
                    overrides["."] = previous
                spawn.prim_physics = overrides
            if spawn.physics_material is None:
                overrides["."] = _ObjectFrictionCfg(friction=friction, previous=previous)
                self._object.object_cfg.spawn = make_usd_spawn_cfg_with_prim_physics(spawn, overrides)
                return

        # Retain configured density, restitution, and backend-specific material properties.
        material = spawn.physics_material
        fragments = list(material) if isinstance(material, (list, tuple)) else []
        if material is not None and not isinstance(material, (list, tuple)):
            fragments.append(material)
        has_friction = False
        for fragment in fragments:
            if isinstance(fragment, (UsdPhysicsRigidBodyMaterialCfg, RigidBodyMaterialBaseCfg)):
                fragment.static_friction = fragment.dynamic_friction = friction
                has_friction = True
        if not has_friction:
            fragments.append(UsdPhysicsRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction))
        spawn.physics_material = fragments[0] if len(fragments) == 1 else fragments
        self._object.object_cfg.spawn = spawn


@configclass
class _ObjectFrictionCfg(UsdPrimSpawnPhysicsCfg):
    """Override friction while inheriting each collider's USD-authored material properties."""

    friction: float = 0.5
    """Coefficient applied to both static and dynamic friction."""

    previous: UsdPrimSpawnPhysicsCfg | None = None
    """Existing root physics override, retained when installing the friction override."""

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Validate the existing root override and reserve collider-local material paths."""
        if self.previous is not None:
            self.previous.validate_target(prim, root)
        for collider in Usd.PrimRange(prim):
            if collider.HasAPI(UsdPhysics.CollisionAPI):
                assert not collider.GetChild(
                    "ArenaFrictionMaterial"
                ), f"Friction material path already exists under {collider.GetPath()}."

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Create local material overrides without changing the shared source materials."""
        if self.previous is not None:
            self.previous.apply(prim, root)
        targets = []
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                original, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial("physics")
                targets.append((child, original))
        for collider, original in targets:
            material = UsdShade.Material.Define(
                prim.GetStage(), collider.GetPath().AppendChild("ArenaFrictionMaterial")
            )
            if original:
                material.GetPrim().GetReferences().AddInternalReference(original.GetPath())
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(self.friction)
            physics.CreateDynamicFrictionAttr(self.friction)
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
