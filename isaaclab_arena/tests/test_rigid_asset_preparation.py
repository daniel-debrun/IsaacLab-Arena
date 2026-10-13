# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
CACHE_DIR_TARGET = "isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir"
PATH_KEYED_OVERRIDES = [
    "prim_physics",
    "rigid_props_fragment",
    "mass_props_fragment_list",
    "mass_props_pattern",
    "visual_material_bindings",
    "relative_visual_material_path",
    "physics_material_prim_path",
]


def _write_asset(
    source_path, body_path, default_path="/Asset", root_scale=1.0, root_translation=None, root_rotate_xyz=None
):
    """Write a rigid cube with an ancestor transform and a sibling material library.

    root_translation and root_rotate_xyz, in degrees, author a translate and a rotateXYZ op on the default prim.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(source_path))
    for prefix in Sdf.Path(body_path).GetPrefixes():
        UsdGeom.Xform.Define(stage, prefix)
    root = stage.GetPrimAtPath(default_path)
    stage.SetDefaultPrim(root)
    root_xformable = UsdGeom.Xformable(root)
    if root_translation is not None:
        root_xformable.AddTranslateOp().Set(Gf.Vec3d(*root_translation))
    if root_rotate_xyz is not None:
        root_xformable.AddRotateXYZOp().Set(Gf.Vec3f(*root_rotate_xyz))
    root_xformable.AddScaleOp().Set(Gf.Vec3f(root_scale))
    body = stage.GetPrimAtPath(body_path)
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Cube.Define(stage, body_path + "/Cube").CreateSizeAttr().Set(2.0)
    if body != root:
        parent = body.GetParent()
        if parent != root:
            UsdGeom.Xformable(parent).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    material = UsdShade.Material.Define(stage, default_path + "/Looks/Material")
    shader = UsdShade.Shader.Define(stage, default_path + "/Looks/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material)
    # Unreferenced top-level content must never enter the relocated asset.
    auxiliary = UsdGeom.Xform.Define(stage, "/Unrelated").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(auxiliary)
    stage.GetRootLayer().Save()
    return stage


def _make_spawn_cfg(source_path, scale=(1.0, 1.0, 1.0), **spawn_options):
    from isaaclab.sim import UsdFileCfg

    return UsdFileCfg(usd_path=str(source_path), scale=scale, **spawn_options)


def _make_spawn_cfg_with_path_keyed_override(override_name, source_path):
    """Return a USD spawn config whose settings name or select prims inside the source asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg

    from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_prim_physics

    usd_path = str(source_path)
    if override_name == "prim_physics":
        return make_usd_spawn_cfg_with_prim_physics(
            sim_utils.UsdFileCfg(usd_path=usd_path), {"Group/Body": UsdPrimSpawnPhysicsCfg()}
        )
    if override_name == "rigid_props_fragment":
        return sim_utils.UsdFileCfg(usd_path=usd_path, rigid_props=UsdPhysicsRigidBodyCfg())
    if override_name == "mass_props_fragment_list":
        return sim_utils.UsdFileCfg(usd_path=usd_path, mass_props=[MassCfg(mass=1.0)])
    if override_name == "mass_props_pattern":
        return sim_utils.UsdFileCfg(usd_path=usd_path, mass_props={"/.*": [MassCfg(mass=1.0)]})
    if override_name == "visual_material_bindings":
        return sim_utils.UsdFileCfg(usd_path=usd_path, visual_material_bindings={"Group/Body/Cube": "Looks/Material"})
    if override_name == "relative_visual_material_path":
        return sim_utils.UsdFileCfg(usd_path=usd_path, visual_material=sim_utils.PreviewSurfaceCfg())
    assert override_name == "physics_material_prim_path", f"Unknown override {override_name}."
    return sim_utils.UsdFileWithCompliantContactCfg(usd_path=usd_path, physics_material_prim_path="Group/Body/Cube")


def _test_matching_body_paths_return_copies_without_writing(simulation_app, tmp_path):
    import torch

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    source_path = tmp_path / "root_body.usda"
    _write_asset(source_path, "/Asset")
    spawn_cfgs = [
        _make_spawn_cfg(source_path, scale=(1.0, 1.0, 1.0)),
        _make_spawn_cfg(source_path, scale=(2.0, 3.0, 4.0), semantic_tags=[("class", "large")]),
        # A primitive with rigid_props has its body at the spawn root, like the USD asset above.
        sim_utils.CuboidCfg(size=(0.1, 0.2, 0.3), rigid_props=sim_utils.RigidBodyPropertiesCfg()),
    ]
    with patch(CACHE_DIR_TARGET) as get_cache_dir:
        aligned = align_rigid_body_paths(spawn_cfgs)
    get_cache_dir.assert_not_called()
    assert [spawn_cfg.usd_path for spawn_cfg in aligned[:2]] == [str(source_path)] * 2
    assert [spawn_cfg.scale for spawn_cfg in aligned[:2]] == [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)]
    assert isinstance(aligned[2], sim_utils.CuboidCfg)
    assert aligned[2].size == (0.1, 0.2, 0.3)
    for aligned_cfg, spawn_cfg in zip(aligned, spawn_cfgs):
        assert aligned_cfg is not spawn_cfg
    aligned[1].semantic_tags.append(("color", "red"))
    assert spawn_cfgs[1].semantic_tags == [("class", "large")]
    assert torch.allclose(SpawnCfgGeometry(aligned[1]).get_bounding_box().size, torch.tensor([[4.0, 6.0, 8.0]]))
    return True


def _test_relocation_preserves_bounds_and_materials(simulation_app, tmp_path):
    import torch

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    source_paths = [
        tmp_path / "root.usda",
        tmp_path / "nested.usda",
        tmp_path / "nested_default.usda",
    ]
    _write_asset(source_paths[0], "/Asset", root_scale=3.0)
    nested_stage = _write_asset(source_paths[1], "/Asset/Group/Body", root_scale=3.0)
    _write_asset(source_paths[2], "/Outer/Asset", default_path="/Outer/Asset", root_scale=3.0)
    geometry_path = tmp_path / "geometry.usda"
    geometry_stage = Usd.Stage.CreateNew(str(geometry_path))
    geometry_root = UsdGeom.Xform.Define(geometry_stage, "/Geometry").GetPrim()
    geometry_stage.SetDefaultPrim(geometry_root)
    UsdGeom.Cube.Define(geometry_stage, "/Geometry/ReferencedCube").CreateSizeAttr().Set(4.0)
    geometry_stage.GetRootLayer().Save()
    nested_stage.GetPrimAtPath("/Asset/Group/Body").GetReferences().AddReference("geometry.usda")
    texture_path = tmp_path / "texture.png"
    texture_path.write_bytes(b"texture-path-test")
    shader = UsdShade.Shader(nested_stage.GetPrimAtPath("/Asset/Looks/Material/Shader"))
    shader.CreateInput("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    inherited_material = UsdShade.Material(nested_stage.GetPrimAtPath("/Asset/Looks/Material"))
    UsdShade.MaterialBindingAPI(nested_stage.GetDefaultPrim()).Bind(
        inherited_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    descendant_material = UsdShade.Material.Define(nested_stage, "/Asset/Looks/DescendantMaterial")
    source_cube = nested_stage.GetPrimAtPath("/Asset/Group/Body/Cube")
    UsdShade.MaterialBindingAPI.Apply(source_cube).Bind(descendant_material)
    assert UsdShade.MaterialBindingAPI(source_cube).ComputeBoundMaterial()[0].GetPath() == inherited_material.GetPath()
    nested_stage.GetRootLayer().Save()
    spawn_cfgs = [_make_spawn_cfg(path, scale=(2.0, 2.0, 2.0)) for path in source_paths]
    original_content = [path.read_bytes() for path in source_paths]
    original_bounds = [SpawnCfgGeometry(spawn_cfg).get_bounding_box() for spawn_cfg in spawn_cfgs]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch(CACHE_DIR_TARGET, return_value=cache_dir):
        aligned = align_rigid_body_paths(spawn_cfgs)
    for source_path, content, spawn_cfg, original_bound in zip(
        source_paths, original_content, aligned, original_bounds
    ):
        assert source_path.read_bytes() == content
        assert spawn_cfg.scale == (2.0, 2.0, 2.0)
        relocated_geometry = SpawnCfgGeometry(spawn_cfg)
        assert relocated_geometry.get_contact_body_path() == "/rigid_body"
        relocated_bound = relocated_geometry.get_bounding_box()
        assert torch.allclose(original_bound.min_point, relocated_bound.min_point)
        assert torch.allclose(original_bound.max_point, relocated_bound.max_point)
        stage = Usd.Stage.CreateInMemory()
        holder = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
        holder.GetReferences().AddReference(spawn_cfg.usd_path)
        bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert [str(body.GetPath()) for body in bodies] == ["/World/Object/rigid_body"]
        material, _ = UsdShade.MaterialBindingAPI(bodies[0]).ComputeBoundMaterial()
        assert material
        connection = material.GetSurfaceOutput().GetAttr().GetConnections()[0]
        assert stage.GetPropertyAtPath(connection)
        if source_path == source_paths[1]:
            assert stage.GetPrimAtPath("/World/Object/rigid_body/ReferencedCube")
            relocated_shader = UsdShade.Shader(stage.GetPrimAtPath(connection.GetPrimPath()))
            assert relocated_shader.GetInput("texture").Get().resolvedPath == str(texture_path)
            relocated_cube = stage.GetPrimAtPath("/World/Object/rigid_body/Cube")
            assert UsdShade.MaterialBindingAPI(relocated_cube).ComputeBoundMaterial()[0].GetPath() == material.GetPath()
    return True


def _test_relocation_redirects_paths_from_outside_into_the_moved_body(simulation_app, tmp_path):
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    nested_stage = _write_asset(nested_path, "/Asset/Group/Body")
    # A prim outside the body that points into it, as joints, filtered pairs and collections do.
    helper = UsdGeom.Xform.Define(nested_stage, "/Asset/Helper").GetPrim()
    helper.CreateRelationship("bodyTargets").SetTargets(["/Asset/Group/Body", "/Asset/Group/Body/Cube"])
    helper_size = helper.CreateAttribute("cubeSize", Sdf.ValueTypeNames.Double)
    helper_size.SetConnections(["/Asset/Group/Body/Cube.size"])
    nested_stage.GetRootLayer().Save()
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths([_make_spawn_cfg(root_path), _make_spawn_cfg(nested_path)])
    stage = Usd.Stage.CreateInMemory()
    holder = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    holder.GetReferences().AddReference(aligned[1].usd_path)
    relocated_helpers = [prim for prim in stage.Traverse() if prim.GetName() == "Helper"]
    assert len(relocated_helpers) == 1
    relocated_helper = relocated_helpers[0]
    assert not relocated_helper.GetPath().HasPrefix("/World/Object/rigid_body")
    body_targets = relocated_helper.GetRelationship("bodyTargets").GetTargets()
    assert body_targets == [Sdf.Path("/World/Object/rigid_body"), Sdf.Path("/World/Object/rigid_body/Cube")]
    for target in body_targets:
        assert stage.GetPrimAtPath(target), target
    size_connections = relocated_helper.GetAttribute("cubeSize").GetConnections()
    assert size_connections == [Sdf.Path("/World/Object/rigid_body/Cube.size")]
    assert stage.GetPropertyAtPath(size_connections[0])
    return True


def _test_relocation_keeps_authored_root_scale_when_spawn_scale_is_none(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset", root_scale=3.0)
    _write_asset(nested_path, "/Asset/Group/Body", root_scale=3.0)
    spawn_cfgs = [_make_spawn_cfg(root_path, scale=None), _make_spawn_cfg(nested_path, scale=None)]
    original_bounds = [SpawnCfgGeometry(spawn_cfg).get_bounding_box() for spawn_cfg in spawn_cfgs]
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths(spawn_cfgs)
    for spawn_cfg, original_bound in zip(aligned, original_bounds):
        assert spawn_cfg.scale is None
        relocated_bound = SpawnCfgGeometry(spawn_cfg).get_bounding_box()
        # The authored root scale of 3 applies exactly once in the relocated copy.
        assert torch.allclose(original_bound.min_point, relocated_bound.min_point)
        assert torch.allclose(original_bound.max_point, relocated_bound.max_point)
    return True


def _test_relocation_drops_the_authored_root_translation_and_rotation(simulation_app, tmp_path):
    import numpy as np

    from isaaclab.sim.utils import use_stage
    from isaaclab.utils.string import string_to_callable
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    root_pose = {"root_translation": (0.4, -0.3, 0.2), "root_rotate_xyz": (90.0, 0.0, 30.0)}
    _write_asset(root_path, "/Asset", **root_pose)
    _write_asset(nested_path, "/Asset/Group/Body", **root_pose)
    spawn_cfgs = [_make_spawn_cfg(root_path), _make_spawn_cfg(nested_path)]
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths(spawn_cfgs)
    # Isaac Lab's spawn replaces the root translation and orientation with the spawn pose, so a body spawned from
    # the source and one spawned from the relocated copy at the same pose must end up at the same world transform.
    spawn_pose = {"translation": (1.0, 2.0, 0.5), "orientation": (0.0, 0.0, 0.3826834, 0.9238795)}
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        for member_index, (spawn_cfg, aligned_cfg) in enumerate(zip(spawn_cfgs, aligned)):
            string_to_callable(spawn_cfg.func)(f"/World/source_{member_index}", spawn_cfg, **spawn_pose)
            string_to_callable(aligned_cfg.func)(f"/World/relocated_{member_index}", aligned_cfg, **spawn_pose)
    source_body_paths = ["/World/source_0", "/World/source_1/Group/Body"]
    for member_index, source_body_path in enumerate(source_body_paths):
        source_body = stage.GetPrimAtPath(source_body_path)
        relocated_body = stage.GetPrimAtPath(f"/World/relocated_{member_index}/rigid_body")
        source_transform = UsdGeom.Xformable(source_body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        relocated_transform = UsdGeom.Xformable(relocated_body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert np.allclose(np.array(source_transform), np.array(relocated_transform), atol=1e-5), source_body_path
    return True


def _write_instanceable_geometry(geometry_path, nested_instance_path=None, rigid_cube=False):
    """Write a geometry layer with a 4 m cube, optionally inside a nested instanceable reference."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(geometry_path))
    root = UsdGeom.Xform.Define(stage, "/Geometry").GetPrim()
    stage.SetDefaultPrim(root)
    if nested_instance_path is None:
        cube = UsdGeom.Cube.Define(stage, "/Geometry/ReferencedCube")
        cube.CreateSizeAttr().Set(4.0)
        if rigid_cube:
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    else:
        inner = UsdGeom.Xform.Define(stage, "/Geometry/Inner").GetPrim()
        inner.GetReferences().AddReference(nested_instance_path.name)
        inner.SetInstanceable(True)
    stage.GetRootLayer().Save()


def _write_asset_with_instanced_visuals(source_path, geometry_path):
    """Write a rigid body at /Asset/Group/Body with an instanceable Visuals child that references geometry_path.

    This is the usual layout of Isaac Sim assets.
    """
    from pxr import UsdGeom

    stage = _write_asset(source_path, "/Asset/Group/Body")
    visuals = UsdGeom.Xform.Define(stage, "/Asset/Group/Body/Visuals").GetPrim()
    visuals.GetReferences().AddReference(geometry_path.name)
    visuals.SetInstanceable(True)
    stage.GetRootLayer().Save()


def _test_relocation_keeps_instanced_geometry_below_the_body(simulation_app, tmp_path, nested_instance):
    import torch

    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths
    from isaaclab_arena.utils.usd.rigid_bodies import get_all_rigid_body_prim_paths_from_stage

    geometry_path = tmp_path / "geometry.usda"
    referenced_cube_path = "Visuals/ReferencedCube"
    if nested_instance:
        inner_geometry_path = tmp_path / "inner_geometry.usda"
        _write_instanceable_geometry(inner_geometry_path)
        _write_instanceable_geometry(geometry_path, nested_instance_path=inner_geometry_path)
        referenced_cube_path = "Visuals/Inner/ReferencedCube"
    else:
        _write_instanceable_geometry(geometry_path)
    root_path = tmp_path / "root.usda"
    instanced_path = tmp_path / "instanced.usda"
    _write_asset(root_path, "/Asset")
    _write_asset_with_instanced_visuals(instanced_path, geometry_path)
    spawn_cfgs = [_make_spawn_cfg(root_path), _make_spawn_cfg(instanced_path)]
    original_bound = SpawnCfgGeometry(spawn_cfgs[1]).get_bounding_box()
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths(spawn_cfgs)
    relocated_bound = SpawnCfgGeometry(aligned[1]).get_bounding_box()
    assert torch.allclose(original_bound.min_point, relocated_bound.min_point)
    assert torch.allclose(original_bound.max_point, relocated_bound.max_point)
    stage = Usd.Stage.CreateInMemory()
    holder = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    holder.GetReferences().AddReference(aligned[1].usd_path)
    assert get_all_rigid_body_prim_paths_from_stage(stage) == ["/World/Object/rigid_body"]
    assert stage.GetPrimAtPath(f"/World/Object/rigid_body/{referenced_cube_path}").IsA(UsdGeom.Cube)
    return True


def _test_relocation_finds_rigid_bodies_inside_instances(simulation_app, tmp_path):
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    geometry_path = tmp_path / "geometry.usda"
    _write_instanceable_geometry(geometry_path, rigid_cube=True)
    root_path = tmp_path / "root.usda"
    instanced_path = tmp_path / "instanced.usda"
    _write_asset(root_path, "/Asset")
    # The second rigid body sits inside an instance, where body discovery does not look.
    _write_asset_with_instanced_visuals(instanced_path, geometry_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch(CACHE_DIR_TARGET, return_value=cache_dir):
        with pytest.raises(AssertionError, match="exactly one rigid body"):
            align_rigid_body_paths([_make_spawn_cfg(root_path), _make_spawn_cfg(instanced_path)])
    assert not list(cache_dir.iterdir())
    return True


def _test_relocated_cache_files_are_named_by_content(simulation_app, tmp_path):
    import re

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    root_path = tmp_path / "root.usda"
    root_copy_path = copy_dir / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    _write_asset(root_copy_path, "/Asset")
    _write_asset(nested_path, "/Asset/Group/Body")
    spawn_cfgs = [_make_spawn_cfg(root_path), _make_spawn_cfg(nested_path, scale=(2.0, 2.0, 2.0))]
    with patch(CACHE_DIR_TARGET, return_value=cache_dir):
        aligned = align_rigid_body_paths(spawn_cfgs)
        first_cache_files = sorted(cache_dir.iterdir())
        modification_times = {path: path.stat().st_mtime_ns for path in first_cache_files}
        repeated = align_rigid_body_paths(spawn_cfgs)
        rescaled = align_rigid_body_paths([spawn_cfgs[0], _make_spawn_cfg(nested_path, scale=(4.0, 4.0, 4.0))])
        from_copy = align_rigid_body_paths([_make_spawn_cfg(root_copy_path), spawn_cfgs[1]])
    assert len(first_cache_files) == 2
    for cache_file in first_cache_files:
        assert re.fullmatch(r"rigid_[0-9a-f]{64}\.usd", cache_file.name), cache_file.name
    assert aligned[0].usd_path != aligned[1].usd_path
    assert [spawn_cfg.usd_path for spawn_cfg in repeated] == [spawn_cfg.usd_path for spawn_cfg in aligned]
    for cache_file in first_cache_files:
        assert cache_file.stat().st_mtime_ns == modification_times[cache_file]
    # The spawn scale stays on the configuration, so it does not change the cache file.
    assert rescaled[1].usd_path == aligned[1].usd_path
    assert rescaled[1].scale == (4.0, 4.0, 4.0)
    # Equal content from another source file shares the cache file, and no temporary files remain.
    assert from_copy[0].usd_path == aligned[0].usd_path
    assert sorted(cache_dir.iterdir()) == first_cache_files
    return True


def _test_relocation_bakes_usd_variant_selections(simulation_app, tmp_path):
    import torch

    from pxr import Usd

    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    variant_path = tmp_path / "variants.usda"
    nested_path = tmp_path / "nested.usda"
    stage = _write_asset(variant_path, "/Asset")
    stage.GetPrimAtPath("/Asset/Cube").GetAttribute("size").Clear()
    variant_set = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("size")
    for variant_name, size in (("small", 1.0), ("large", 4.0)):
        variant_set.AddVariant(variant_name)
        variant_set.SetVariantSelection(variant_name)
        with variant_set.GetVariantEditContext():
            stage.GetPrimAtPath("/Asset/Cube").GetAttribute("size").Set(size)
    variant_set.SetVariantSelection("small")
    stage.GetRootLayer().Save()
    _write_asset(nested_path, "/Asset/Group/Body")
    source_content = variant_path.read_bytes()
    spawn_cfgs = [_make_spawn_cfg(variant_path, variants={"size": "large"}), _make_spawn_cfg(nested_path)]
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths(spawn_cfgs)
    assert aligned[0].variants is None
    assert spawn_cfgs[0].variants == {"size": "large"}
    assert torch.allclose(SpawnCfgGeometry(aligned[0]).get_bounding_box().size, torch.tensor([[4.0, 4.0, 4.0]]))
    relocated_stage = Usd.Stage.Open(aligned[0].usd_path)
    for prim in relocated_stage.Traverse():
        assert not prim.GetVariantSets().GetNames(), prim.GetPath()
    assert variant_path.read_bytes() == source_content
    return True


def _test_relocation_rejects_path_keyed_overrides(simulation_app, tmp_path, override_name):
    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    _write_asset(nested_path, "/Asset/Group/Body")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    spawn_cfgs = [_make_spawn_cfg(root_path), _make_spawn_cfg_with_path_keyed_override(override_name, nested_path)]
    with patch(CACHE_DIR_TARGET, return_value=cache_dir):
        with pytest.raises(AssertionError, match="Cannot relocate"):
            align_rigid_body_paths(spawn_cfgs)
    # Every member is checked before the first file is written.
    assert not list(cache_dir.iterdir())
    return True


def _test_legacy_physics_cfgs_apply_to_relocated_body(simulation_app, tmp_path):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import use_stage
    from isaaclab.utils.string import string_to_callable
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    root_stage = _write_asset(root_path, "/Asset")
    nested_stage = _write_asset(nested_path, "/Asset/Group/Body")
    # Isaac Lab's legacy mass properties only edit bodies that already carry MassAPI.
    UsdPhysics.MassAPI.Apply(root_stage.GetPrimAtPath("/Asset"))
    root_stage.GetRootLayer().Save()
    UsdPhysics.MassAPI.Apply(nested_stage.GetPrimAtPath("/Asset/Group/Body"))
    nested_stage.GetRootLayer().Save()
    expected_masses = [2.0, 3.0]
    spawn_cfgs = []
    for source_path, mass in zip((root_path, nested_path), expected_masses):
        spawn_cfgs.append(
            _make_spawn_cfg(
                source_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            )
        )
    with patch(CACHE_DIR_TARGET, return_value=tmp_path):
        aligned = align_rigid_body_paths(spawn_cfgs)
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        for member_index, spawn_cfg in enumerate(aligned):
            spawn_function = string_to_callable(spawn_cfg.func)
            spawn_function(f"/World/member_{member_index}", spawn_cfg)
    for member_index, expected_mass in enumerate(expected_masses):
        body = stage.GetPrimAtPath(f"/World/member_{member_index}/rigid_body")
        assert UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(expected_mass)
    return True


def _test_differing_body_paths_require_usd_members(simulation_app, tmp_path):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    nested_path = tmp_path / "nested.usda"
    _write_asset(nested_path, "/Asset/Group/Body")
    spawn_cfgs = [
        sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), rigid_props=sim_utils.RigidBodyPropertiesCfg()),
        _make_spawn_cfg(nested_path),
    ]
    with patch(CACHE_DIR_TARGET) as get_cache_dir:
        with pytest.raises(AssertionError, match="must be USD assets"):
            align_rigid_body_paths(spawn_cfgs)
    get_cache_dir.assert_not_called()
    return True


def _test_relocation_requires_exactly_one_rigid_body(simulation_app, tmp_path):
    from pxr import UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths

    two_body_path = tmp_path / "two_bodies.usda"
    nested_path = tmp_path / "nested.usda"
    stage = _write_asset(two_body_path, "/Asset")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Asset/Extra").GetPrim())
    stage.GetRootLayer().Save()
    _write_asset(nested_path, "/Asset/Group/Body")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch(CACHE_DIR_TARGET, return_value=cache_dir):
        # Matching paths take the fast path, which keeps main's shallowest-body rule.
        aligned = align_rigid_body_paths([_make_spawn_cfg(two_body_path), _make_spawn_cfg(two_body_path)])
        assert [spawn_cfg.usd_path for spawn_cfg in aligned] == [str(two_body_path)] * 2
        # In the second order the nested member relocates fine before the two-body member is rejected.
        for member_paths in ((two_body_path, nested_path), (nested_path, two_body_path)):
            with pytest.raises(AssertionError, match="exactly one rigid body"):
                align_rigid_body_paths([_make_spawn_cfg(path) for path in member_paths])
    # Every relocated layer is built before the first file is written.
    assert not list(cache_dir.iterdir())
    return True


def test_matching_body_paths_return_copies_without_writing(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_matching_body_paths_return_copies_without_writing, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_preserves_bounds_and_materials(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_preserves_bounds_and_materials, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_redirects_paths_from_outside_into_the_moved_body(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_redirects_paths_from_outside_into_the_moved_body, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_keeps_authored_root_scale_when_spawn_scale_is_none(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_keeps_authored_root_scale_when_spawn_scale_is_none, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_drops_the_authored_root_translation_and_rotation(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_drops_the_authored_root_translation_and_rotation, headless=HEADLESS, tmp_path=tmp_path
    )


@pytest.mark.parametrize("nested_instance", [False, True])
def test_relocation_keeps_instanced_geometry_below_the_body(tmp_path, nested_instance):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_keeps_instanced_geometry_below_the_body,
        headless=HEADLESS,
        tmp_path=tmp_path,
        nested_instance=nested_instance,
    )


def test_relocation_finds_rigid_bodies_inside_instances(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_finds_rigid_bodies_inside_instances, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocated_cache_files_are_named_by_content(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocated_cache_files_are_named_by_content, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_bakes_usd_variant_selections(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_bakes_usd_variant_selections, headless=HEADLESS, tmp_path=tmp_path
    )


@pytest.mark.parametrize("override_name", PATH_KEYED_OVERRIDES)
def test_relocation_rejects_path_keyed_overrides(tmp_path, override_name):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_rejects_path_keyed_overrides,
        headless=HEADLESS,
        tmp_path=tmp_path,
        override_name=override_name,
    )


def test_legacy_physics_cfgs_apply_to_relocated_body(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_legacy_physics_cfgs_apply_to_relocated_body, headless=HEADLESS, tmp_path=tmp_path
    )


def test_differing_body_paths_require_usd_members(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_differing_body_paths_require_usd_members, headless=HEADLESS, tmp_path=tmp_path
    )


def test_relocation_requires_exactly_one_rigid_body(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_relocation_requires_exactly_one_rigid_body, headless=HEADLESS, tmp_path=tmp_path
    )
