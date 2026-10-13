# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Move rigid bodies to one common prim path, so several assets can spawn as one Isaac Lab rigid object.

Isaac Lab finds a rigid object's body in the first environment and looks for it at the same path below the spawn
root in every other environment. When the environments spawn assets that keep their rigid body at different paths,
a copy of each USD file is written with the body at /rigid_body.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from functools import partial
from typing import Any, NamedTuple

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas import SchemaFragment
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import make_uninstanceable, standardize_xform_ops
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
from isaaclab_arena.utils.usd.rigid_bodies import get_all_rigid_body_prim_paths_from_stage

_SOURCE_ROOT = Sdf.Path("/Asset")
"""Where SpawnCfgGeometry.open_usd_stage() references the source asset."""

_COPY_ROOT = Sdf.Path("/Prepared")
"""Default prim of a written copy."""

_COPY_REST_OF_ASSET = _COPY_ROOT.AppendChild("source")
"""Everything from the source asset except its rigid body."""

_COPY_BODY = _COPY_ROOT.AppendChild("rigid_body")
"""The moved rigid body, at /rigid_body relative to the spawn root."""

_CACHE_LAYOUT_VERSION = "rigid-layout-v1"
"""Part of every cache hash. Change it when the layout of the written copies changes."""


class _BoundMaterial(NamedTuple):
    """A material bound to the rigid body in the source asset, directly or through a parent prim."""

    purpose: str
    """Material purpose of the binding, such as all-purpose or preview."""

    material_path: Sdf.Path
    """Path of the material in the source stage."""

    binding_strength: str
    """Binding strength of the relationship that binds the material."""


def align_rigid_body_paths(spawn_cfgs: Sequence[SpawnerCfg]) -> list[SpawnerCfg]:
    """Return spawn configurations whose rigid bodies sit at the same path inside their assets.

    If they already do, the configurations are returned as copies. Otherwise each USD file is copied with its rigid
    body moved to /rigid_body, and each returned configuration points at its copy, with its USD variant selections
    baked in.

    Args:
        spawn_cfgs: Native spawn configurations.

    Returns:
        The aligned spawn configurations, in input order.
    """
    assert spawn_cfgs, "At least one spawn configuration is required."
    geometries = [SpawnCfgGeometry(spawn_cfg) for spawn_cfg in spawn_cfgs]
    body_paths = [geometry.get_contact_body_path() for geometry in geometries]
    if len(set(body_paths)) == 1:
        return [deepcopy(spawn_cfg) for spawn_cfg in spawn_cfgs]
    for spawn_cfg in spawn_cfgs:
        assert isinstance(spawn_cfg, UsdFileCfg), (
            "Spawn configurations with different rigid-body paths must be USD assets, so their bodies can be moved "
            f"to one path; got {type(spawn_cfg).__name__} and rigid-body paths {body_paths}."
        )
        _assert_settings_do_not_name_asset_prims(spawn_cfg)
    # Build every copy before saving any of them, so a rejected spawn configuration leaves no cache files behind.
    copied_layers = [
        _build_copy_with_body_moved(geometry, body_path) for geometry, body_path in zip(geometries, body_paths)
    ]
    aligned_spawn_cfgs = []
    for spawn_cfg, copied_layer in zip(spawn_cfgs, copied_layers):
        aligned_spawn_cfg = deepcopy(spawn_cfg)
        aligned_spawn_cfg.usd_path = str(_save_to_cache(copied_layer))
        # The copy has the selections baked in and has no variant sets of its own.
        aligned_spawn_cfg.variants = None
        aligned_spawn_cfgs.append(aligned_spawn_cfg)
    return aligned_spawn_cfgs


def _assert_settings_do_not_name_asset_prims(spawn_cfg: UsdFileCfg) -> None:
    """Reject spawn settings that name or select prims inside the source asset, because those prims move in the copy."""
    cannot_move = f"Cannot move the rigid body of {spawn_cfg.usd_path} to a common path"
    assert not getattr(spawn_cfg, "prim_physics", None), f"{cannot_move}: prim_physics names source prims."
    for spawn_field in fields(spawn_cfg):
        if not spawn_field.name.endswith("_props"):
            continue
        assert not _names_prims_inside_asset(getattr(spawn_cfg, spawn_field.name)), (
            f"{cannot_move}: {spawn_field.name} uses schema fragments or target patterns. "
            "Use a properties config such as MassPropertiesCfg, or spawn configurations with the same rigid-body path."
        )
    assert not spawn_cfg.visual_material_bindings, f"{cannot_move}: visual_material_bindings name source prims."
    # Isaac Lab reads visual_material_path only when visual_material is set.
    assert spawn_cfg.visual_material is None or spawn_cfg.visual_material_path.startswith(
        "/"
    ), f"{cannot_move}: visual_material_path is relative to the asset root."
    assert (
        getattr(spawn_cfg, "physics_material_prim_path", None) is None
    ), f"{cannot_move}: physics_material_prim_path names source prims."


def _names_prims_inside_asset(props: Any) -> bool:
    """Return whether a ``*_props`` value is a schema fragment, a list of fragments or a target-pattern mapping."""
    if isinstance(props, SchemaFragment):
        return True
    if isinstance(props, dict):
        return bool(props)
    if isinstance(props, (list, tuple)):
        return any(isinstance(item, SchemaFragment) for item in props)
    return False


def _build_copy_with_body_moved(geometry: SpawnCfgGeometry, body_path: str) -> Sdf.Layer:
    """Return a flattened copy of one USD asset with its rigid body moved to /rigid_body.

    Args:
        geometry: Geometry of a UsdFileCfg spawn configuration.
        body_path: Its rigid-body path relative to the default prim, "" for the default prim.

    Returns:
        The copy as a layer, not yet saved to the cache.
    """
    stage = _open_asset_for_editing(geometry)
    # The body has three paths here: source_body_path under /Asset in the source stage, body_path_in_rest_of_asset
    # under /Prepared/source in the copy, and finally /Prepared/rigid_body.
    source_body_path = Sdf.Path(str(_SOURCE_ROOT) + body_path)
    _assert_single_rigid_body(stage, source_body_path, geometry.spawn_cfg.usd_path)

    # Read these while the body still sits below its parents. Once moved, it no longer inherits their transforms
    # or material bindings, so both are authored on the moved body directly.
    body = stage.GetPrimAtPath(source_body_path)
    # /Asset has no parent, so the body's world transform is its transform within the asset.
    body_transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    bound_materials = _read_bound_materials(body)

    prepared_stage, body_path_in_rest_of_asset = _move_body_to_common_path(stage, source_body_path)
    _restore_placement_and_materials(prepared_stage, body_transform, bound_materials, body_path_in_rest_of_asset)
    return prepared_stage.GetRootLayer()


def _open_asset_for_editing(geometry: SpawnCfgGeometry) -> Usd.Stage:
    """Open the USD asset in a throwaway in-memory stage, editable and with the spawn root's placement cleared."""
    stage = geometry.open_usd_stage()
    # Instance proxies cannot be edited or found by Traverse(), so make the whole asset editable before looking
    # for bodies.
    make_uninstanceable(_SOURCE_ROOT, stage=stage)
    # Isaac Lab sets the spawn root's translation and orientation, and a tuple spawn scale replaces its scale.
    # Clear them here, so the moved body keeps only its transform relative to the spawn root.
    standardize_xform_ops(
        stage.GetDefaultPrim(),
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0) if geometry.spawn_cfg.scale is not None else None,
    )
    return stage


def _assert_single_rigid_body(stage: Usd.Stage, body_path: Sdf.Path, usd_path: str) -> None:
    """Assert that the stage's only rigid body is at ``body_path``."""
    rigid_body_paths = get_all_rigid_body_prim_paths_from_stage(stage)
    assert rigid_body_paths == [
        str(body_path)
    ], f"Moving the rigid body of {usd_path} requires exactly one rigid body; found {rigid_body_paths}."


def _read_bound_materials(body: Usd.Prim) -> list[_BoundMaterial]:
    """Return the materials bound to the body for each material purpose, including bindings on its parents."""
    bound_materials = []
    for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
        material, relationship = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial(purpose)
        if material:
            binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
            bound_materials.append(_BoundMaterial(purpose, material.GetPath(), binding_strength))
    return bound_materials


def _move_body_to_common_path(stage: Usd.Stage, source_body_path: Sdf.Path) -> tuple[Usd.Stage, Sdf.Path]:
    """Flatten the asset into a new stage with its rigid body moved to /Prepared/rigid_body.

    The rest of the asset goes to /Prepared/source, and properties that pointed into the body point at its new path.

    Args:
        stage: Source stage with the asset at /Asset.
        source_body_path: Path of the rigid body below /Asset in the source stage.

    Returns:
        A stage on the new layer, and the path the body's copy had below /Prepared/source before it moved.
    """
    # Flatten resolves references and USD variant selections and makes asset paths absolute, so the copy can
    # live in another directory.
    layer = stage.Flatten(addSourceFileComment=False)
    prepared_stage = Usd.Stage.Open(layer)
    prepared_root = UsdGeom.Xform.Define(prepared_stage, _COPY_ROOT).GetPrim()
    # Keep the rest of the source asset: materials and other content the body uses may sit outside the body.
    copied_rest_of_asset = Sdf.CopySpec(layer, _SOURCE_ROOT, layer, _COPY_REST_OF_ASSET)
    assert copied_rest_of_asset, f"Could not copy {_SOURCE_ROOT} to {_COPY_REST_OF_ASSET}."
    body_path_in_rest_of_asset = source_body_path.ReplacePrefix(_SOURCE_ROOT, _COPY_REST_OF_ASSET)
    copied_body = Sdf.CopySpec(layer, body_path_in_rest_of_asset, layer, _COPY_BODY)
    assert copied_body, f"Could not copy {body_path_in_rest_of_asset} to {_COPY_BODY}."
    prepared_stage.RemovePrim(body_path_in_rest_of_asset)
    prepared_stage.RemovePrim(_SOURCE_ROOT)
    prepared_stage.SetDefaultPrim(prepared_root)
    _point_links_at_moved_body(layer, body_path_in_rest_of_asset)
    return prepared_stage, body_path_in_rest_of_asset


def _point_links_at_moved_body(layer: Sdf.Layer, body_path_in_rest_of_asset: Sdf.Path) -> None:
    """Point relationship targets and attribute connections into body_path_in_rest_of_asset at the moved body."""
    # Sdf.CopySpec remaps only paths inside the subtree it copies, so prims outside the body that point into it,
    # such as joints, filtered pairs or collections, still name the removed body below /Prepared/source.
    path_after_body_move = partial(_path_after_body_move, body_path_in_rest_of_asset=body_path_in_rest_of_asset)
    property_paths = []

    def collect_property_path(spec_path: Sdf.Path) -> None:
        if spec_path.IsPropertyPath():
            property_paths.append(spec_path)

    # Collect the paths first, so the list edits below never run while the layer is being traversed.
    layer.Traverse(Sdf.Path.absoluteRootPath, collect_property_path)
    for property_path in property_paths:
        property_spec = layer.GetPropertyAtPath(property_path)
        if isinstance(property_spec, Sdf.RelationshipSpec):
            property_spec.targetPathList.ModifyItemEdits(path_after_body_move)
        elif isinstance(property_spec, Sdf.AttributeSpec):
            property_spec.connectionPathList.ModifyItemEdits(path_after_body_move)


def _path_after_body_move(path: Sdf.Path, body_path_in_rest_of_asset: Sdf.Path) -> Sdf.Path:
    """Return where a path in the copy points after the body moved to /Prepared/rigid_body.

    Paths at or below body_path_in_rest_of_asset, below /Prepared/source, map into /Prepared/rigid_body; others are
    unchanged.
    """
    if path.HasPrefix(body_path_in_rest_of_asset):
        return path.ReplacePrefix(body_path_in_rest_of_asset, _COPY_BODY)
    return path


def _restore_placement_and_materials(
    prepared_stage: Usd.Stage,
    body_transform: Gf.Matrix4d,
    bound_materials: list[_BoundMaterial],
    body_path_in_rest_of_asset: Sdf.Path,
) -> None:
    """Author the body's transform and material bindings from the source asset on the moved body.

    Args:
        prepared_stage: Stage with the moved body at /Prepared/rigid_body.
        body_transform: The body's transform within the source asset.
        bound_materials: Materials bound to the body in the source asset.
        body_path_in_rest_of_asset: The path the body's copy had below /Prepared/source before it moved.
    """
    moved_body = prepared_stage.GetPrimAtPath(_COPY_BODY)
    UsdGeom.Xformable(moved_body).MakeMatrixXform().Set(body_transform)
    for bound_material in bound_materials:
        material_path_in_copy = _path_after_body_move(
            bound_material.material_path.ReplacePrefix(_SOURCE_ROOT, _COPY_REST_OF_ASSET), body_path_in_rest_of_asset
        )
        material = UsdShade.Material(prepared_stage.GetPrimAtPath(material_path_in_copy))
        UsdShade.MaterialBindingAPI.Apply(moved_body).Bind(
            material, bindingStrength=bound_material.binding_strength, materialPurpose=bound_material.purpose
        )


def _save_to_cache(layer: Sdf.Layer) -> pathlib.Path:
    """Save the layer to a cache file named by a hash of its content, and return the file path.

    A file that already exists is reused and not written again.
    """
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
