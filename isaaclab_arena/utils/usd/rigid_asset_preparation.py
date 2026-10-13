# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Give the spawn configurations of one multi-asset object a common rigid-body path."""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from functools import partial
from typing import Any

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas import SchemaFragment
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import make_uninstanceable, standardize_xform_ops
from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
from isaaclab_arena.utils.usd.rigid_bodies import get_all_rigid_body_prim_paths_from_stage

_SOURCE_ROOT = Sdf.Path("/Asset")
"""Where SpawnCfgGeometry.open_usd_stage() references the source asset."""

_PREPARED_ROOT = Sdf.Path("/Prepared")
"""Default prim of a relocated asset."""

_PREPARED_SOURCE = _PREPARED_ROOT.AppendChild("source")
"""The source asset without its rigid body."""

_PREPARED_BODY = _PREPARED_ROOT.AppendChild("rigid_body")
"""The moved rigid body, at /rigid_body relative to the spawn root."""

_CACHE_LAYOUT_VERSION = "rigid-layout-v1"
"""Part of every cache hash. Change it when the relocated layout changes."""


def align_rigid_body_paths(spawn_cfgs: Sequence[SpawnerCfg]) -> list[SpawnerCfg]:
    """Return copies of the spawn configurations with their rigid bodies at one common path.

    When the body paths already agree, the copies are unchanged and nothing is written. Otherwise every member
    must be a USD asset and gets a cached copy with its body at /rigid_body and its USD variant selections
    baked in.

    Args:
        spawn_cfgs: One native spawn configuration per member, in member order.

    Returns:
        One spawn configuration per member, in the same order.
    """
    assert spawn_cfgs, "At least one spawn configuration is required."
    geometries = [SpawnCfgGeometry(spawn_cfg) for spawn_cfg in spawn_cfgs]
    body_paths = [geometry.get_contact_body_path() for geometry in geometries]
    if len(set(body_paths)) == 1:
        return [deepcopy(spawn_cfg) for spawn_cfg in spawn_cfgs]
    for spawn_cfg in spawn_cfgs:
        assert isinstance(spawn_cfg, UsdFileCfg), (
            "Members with different rigid-body paths must be USD assets, so their bodies can be moved to one path; "
            f"got {type(spawn_cfg).__name__} and rigid-body paths {body_paths}."
        )
        _assert_relocatable(spawn_cfg)
    # Build every relocated layer before writing any file, so a rejected member leaves no cache files behind.
    relocated_layers = [
        _relocate_rigid_body(geometry, body_path) for geometry, body_path in zip(geometries, body_paths)
    ]
    aligned_spawn_cfgs = []
    for spawn_cfg, relocated_layer in zip(spawn_cfgs, relocated_layers):
        aligned_spawn_cfg = deepcopy(spawn_cfg)
        aligned_spawn_cfg.usd_path = str(_write_layer_to_cache(relocated_layer))
        # The copy has the selections baked in and has no variant sets of its own.
        aligned_spawn_cfg.variants = None
        aligned_spawn_cfgs.append(aligned_spawn_cfg)
    return aligned_spawn_cfgs


def _assert_relocatable(spawn_cfg: UsdFileCfg) -> None:
    """Reject spawn settings that name or select prims inside the source asset, whose layout relocation changes."""
    source = spawn_cfg.usd_path
    assert not getattr(spawn_cfg, "prim_physics", None), f"Cannot relocate {source}: prim_physics names source prims."
    for spawn_field in fields(spawn_cfg):
        if not spawn_field.name.endswith("_props"):
            continue
        assert not _selects_prims_by_path(getattr(spawn_cfg, spawn_field.name)), (
            f"Cannot relocate {source}: {spawn_field.name} uses schema fragments or target patterns. "
            "Use a properties config such as MassPropertiesCfg, or members with the same rigid-body path."
        )
    assert (
        not spawn_cfg.visual_material_bindings
    ), f"Cannot relocate {source}: visual_material_bindings name source prims."
    # Isaac Lab reads visual_material_path only when visual_material is set.
    assert spawn_cfg.visual_material is None or spawn_cfg.visual_material_path.startswith(
        "/"
    ), f"Cannot relocate {source}: visual_material_path is relative to the asset root."
    assert (
        getattr(spawn_cfg, "physics_material_prim_path", None) is None
    ), f"Cannot relocate {source}: physics_material_prim_path names source prims."


def _selects_prims_by_path(props: Any) -> bool:
    """Return whether a ``*_props`` value is a schema fragment, a list of fragments or a target-pattern mapping."""
    if isinstance(props, SchemaFragment):
        return True
    if isinstance(props, dict):
        return bool(props)
    if isinstance(props, (list, tuple)):
        return any(isinstance(item, SchemaFragment) for item in props)
    return False


def _redirect_into_moved_body(path: Sdf.Path, copied_body: Sdf.Path) -> Sdf.Path:
    """Map a path below the body's copy in /Prepared/source to /Prepared/rigid_body; return other paths unchanged."""
    if path.HasPrefix(copied_body):
        return path.ReplacePrefix(copied_body, _PREPARED_BODY)
    return path


def _redirect_paths_into_moved_body(layer: Sdf.Layer, copied_body: Sdf.Path) -> None:
    """Point relationship targets and attribute connections that name the body's old path at /Prepared/rigid_body.

    Sdf.CopySpec remaps only paths inside the subtree it copies, so prims outside the body that point into it,
    such as joints, filtered pairs or collections, still name the removed body copy below /Prepared/source.
    """
    redirect_into_moved_body = partial(_redirect_into_moved_body, copied_body=copied_body)
    property_paths = []

    def collect_property_path(spec_path: Sdf.Path) -> None:
        if spec_path.IsPropertyPath():
            property_paths.append(spec_path)

    # Collect the paths first, so the list edits below never run while the layer is being traversed.
    layer.Traverse(Sdf.Path.absoluteRootPath, collect_property_path)
    for property_path in property_paths:
        property_spec = layer.GetPropertyAtPath(property_path)
        if isinstance(property_spec, Sdf.RelationshipSpec):
            property_spec.targetPathList.ModifyItemEdits(redirect_into_moved_body)
        elif isinstance(property_spec, Sdf.AttributeSpec):
            property_spec.connectionPathList.ModifyItemEdits(redirect_into_moved_body)


def _relocate_rigid_body(geometry: SpawnCfgGeometry, body_path: str) -> Sdf.Layer:
    """Return a flattened copy of one USD asset with its rigid body moved to /rigid_body.

    Args:
        geometry: Geometry of the member's spawn configuration, which must be a UsdFileCfg.
        body_path: The member's rigid-body path relative to its default prim, "" for the default prim.

    Returns:
        The relocated layer, not yet written to the cache.
    """
    spawn_cfg = geometry.spawn_cfg
    stage = geometry.open_usd_stage()
    # Instance proxies cannot be edited or found by Traverse(), so make the whole asset editable in this
    # throwaway stage before looking for bodies.
    make_uninstanceable(_SOURCE_ROOT, stage=stage)
    original_body = Sdf.Path(str(_SOURCE_ROOT) + body_path)
    rigid_body_paths = get_all_rigid_body_prim_paths_from_stage(stage)
    assert rigid_body_paths == [
        str(original_body)
    ], f"Moving the rigid body of {spawn_cfg.usd_path} requires exactly one rigid body; found {rigid_body_paths}."
    body = stage.GetPrimAtPath(original_body)

    # Isaac Lab sets the spawn root's translation and orientation, and a tuple spawn scale replaces its scale.
    # Clear them here, so the moved body keeps only its transform relative to the spawn root.
    standardize_xform_ops(
        stage.GetDefaultPrim(),
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0) if spawn_cfg.scale is not None else None,
    )
    body_transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inherited_materials = []
    for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
        material, relationship = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial(purpose)
        if material:
            binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
            inherited_materials.append((purpose, material.GetPath(), binding_strength))

    # Flatten resolves references and USD variant selections and makes asset paths absolute, so the copy can
    # live in another directory. Keep the rest of the source asset: materials and other content the body uses
    # may sit outside the body.
    layer = stage.Flatten(addSourceFileComment=False)
    prepared_stage = Usd.Stage.Open(layer)
    prepared_root = UsdGeom.Xform.Define(prepared_stage, _PREPARED_ROOT).GetPrim()
    assert Sdf.CopySpec(layer, _SOURCE_ROOT, layer, _PREPARED_SOURCE)
    copied_body = original_body.ReplacePrefix(_SOURCE_ROOT, _PREPARED_SOURCE)
    assert Sdf.CopySpec(layer, copied_body, layer, _PREPARED_BODY)
    prepared_stage.RemovePrim(copied_body)
    prepared_stage.RemovePrim(_SOURCE_ROOT)
    prepared_stage.SetDefaultPrim(prepared_root)
    _redirect_paths_into_moved_body(layer, copied_body)

    prepared_body = prepared_stage.GetPrimAtPath(_PREPARED_BODY)
    UsdGeom.Xformable(prepared_body).MakeMatrixXform().Set(body_transform)
    for purpose, material_path, binding_strength in inherited_materials:
        prepared_material_path = _redirect_into_moved_body(
            material_path.ReplacePrefix(_SOURCE_ROOT, _PREPARED_SOURCE), copied_body
        )
        material = UsdShade.Material(prepared_stage.GetPrimAtPath(prepared_material_path))
        UsdShade.MaterialBindingAPI.Apply(prepared_body).Bind(
            material, bindingStrength=binding_strength, materialPurpose=purpose
        )
    return layer


def _write_layer_to_cache(layer: Sdf.Layer) -> pathlib.Path:
    """Export the layer once to a cache file named by a hash of its content, and return the file path."""
    cache_identity = f"{_CACHE_LAYOUT_VERSION}\n{layer.ExportToString()}"
    content_hash = hashlib.sha256(cache_identity.encode()).hexdigest()
    cache_path = get_arena_asset_cache_dir() / f"rigid_{content_hash}.usd"
    if cache_path.exists():
        return cache_path
    # Export to a temporary file and rename it, so another process never reads a half-written file.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".usd", delete=False) as temporary_file:
            temporary_path = temporary_file.name
        layer.Export(temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return cache_path
