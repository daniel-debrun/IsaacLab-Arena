# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contact material presets applied to selected colliders during USD spawning."""

import math
from dataclasses import field

from isaaclab.utils.configclass import configclass
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg


@configclass
class ContactMaterialCfg:
    """Friction and restitution coefficients for a named contact material."""

    static_friction: float = 0.5
    """Nonnegative, dimensionless static friction coefficient."""

    dynamic_friction: float = 0.4
    """Nonnegative dynamic friction coefficient, no greater than static friction."""

    restitution: float = 0.0
    """Coefficient of restitution in [0, 1]."""

    def validate_coefficients(self) -> None:
        """Require finite, physically consistent contact coefficients."""
        assert (
            math.isfinite(self.static_friction) and self.static_friction >= 0
        ), "Static friction must be finite and nonnegative."
        assert (
            math.isfinite(self.dynamic_friction) and 0 <= self.dynamic_friction <= self.static_friction
        ), "Dynamic friction must be finite, nonnegative, and no greater than static friction."
        assert math.isfinite(self.restitution) and 0 <= self.restitution <= 1, "Restitution must be in [0, 1]."


@configclass
class ColliderContactMaterialCfg(UsdPrimSpawnPhysicsCfg):
    """Override contact coefficients locally, preserving other bound material properties."""

    material: ContactMaterialCfg = field(default_factory=ContactMaterialCfg)
    """Coefficients applied to this collider."""

    friction_combine_mode: str = "average"
    """PhysX contact combine rule: average, min, multiply, or max; ignored by Newton."""

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Require a collider whose effective material binding can be replaced locally."""
        self.material.validate_coefficients()
        assert self.friction_combine_mode in {"average", "min", "multiply", "max"}, "Invalid friction combine mode."
        assert prim.HasAPI(UsdPhysics.CollisionAPI), f"Contact material target must be a collider: {prim.GetPath()}"
        assert not prim.GetStage().GetPrimAtPath(
            prim.GetPath().AppendChild("ArenaContactMaterial")
        ), f"Contact material path already exists under {prim.GetPath()}."
        _, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        # A physics-purpose binding takes precedence over an all-purpose fallback.
        binding_name = binding.GetName() if binding else ""
        if binding_name == "material:binding:physics" or binding_name.startswith(
            "material:binding:collection:physics:"
        ):
            if binding.GetPrim() == prim:
                assert not binding.GetName().startswith(
                    "material:binding:collection:"
                ), f"A collection material binding would shadow the contact material at {prim.GetPath()}."
            else:
                assert UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(binding) != (
                    UsdShade.Tokens.strongerThanDescendants
                ), f"An ancestor material binding would shadow the contact material at {prim.GetPath()}."

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Author a physics-only binding that remaps with the collider during cloning."""
        original, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        material = UsdShade.Material.Define(prim.GetStage(), prim.GetPath().AppendChild("ArenaContactMaterial"))
        if original:
            # Inherit density, compliance, and other unvaried properties without editing the
            # shared material. Local opinions below override only the selected coefficients.
            material.GetPrim().GetReferences().AddInternalReference(original.GetPath())
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(self.material.static_friction)
        physics.CreateDynamicFrictionAttr(self.material.dynamic_friction)
        physics.CreateRestitutionAttr(self.material.restitution)
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateFrictionCombineModeAttr(self.friction_combine_mode)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
