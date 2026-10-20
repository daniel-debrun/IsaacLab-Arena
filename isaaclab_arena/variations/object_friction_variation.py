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

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
    UsdPhysicsRigidBodyMaterialCfg,
)
from isaaclab.utils.configclass import configclass

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
