# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for geometry derived from one native spawn configuration."""

from __future__ import annotations

import torch
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _write_nested_body_usd(path: Path) -> None:
    """Write /Asset (root scale 0.5) > Group (translated) > Body (rigid, translated, rotated) > Cube."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
    group = UsdGeom.Xform.Define(stage, "/Asset/Group")
    group.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    body = UsdGeom.Xform.Define(stage, "/Asset/Group/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    body.AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/Group/Body/Cube")
    cube.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 3.0))
    stage.GetRootLayer().Save()


def _write_root_scaled_body_usd(path: Path) -> None:
    """Write a rigid /Root with an authored 0.5 root scale over a 2 m cube."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()


def _write_size_variant_usd(path: Path) -> None:
    """Write a cube whose size comes from variant set 'size' (small=1, large=4), selected 'small'."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/Cube")
    size_set = root.GetPrim().GetVariantSets().AddVariantSet("size")
    for name, size in (("small", 1.0), ("large", 4.0)):
        size_set.AddVariant(name)
        size_set.SetVariantSelection(name)
        with size_set.GetVariantEditContext():
            cube.GetSizeAttr().Set(size)
    size_set.SetVariantSelection("small")
    stage.GetRootLayer().Save()


def _write_layout_variant_usd(path: Path) -> None:
    """Write variant set 'layout' whose 'first'/'second' variants each define one rigid body."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    layout_set = root.GetPrim().GetVariantSets().AddVariantSet("layout")
    for name, body_name in (("first", "FirstBody"), ("second", "SecondBody")):
        layout_set.AddVariant(name)
        layout_set.SetVariantSelection(name)
        with layout_set.GetVariantEditContext():
            body = UsdGeom.Xform.Define(stage, f"/Asset/{body_name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdGeom.Cube.Define(stage, f"/Asset/{body_name}/Cube")
    layout_set.SetVariantSelection("first")
    stage.GetRootLayer().Save()


def _test_usd_bounds_and_contact_body_match_main_helpers(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd
    from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body

    usd_path = asset_dir / "nested.usda"
    _write_nested_body_usd(usd_path)
    original_content = usd_path.read_bytes()

    geometry = ObjectGeometry(UsdFileCfg(usd_path=str(usd_path), scale=(2.0, 2.0, 2.0)))
    expected = compute_local_bounding_box_from_usd(str(usd_path), (2.0, 2.0, 2.0))
    assert torch.allclose(geometry.get_bounding_box().min_point, expected.min_point)
    assert torch.allclose(geometry.get_bounding_box().max_point, expected.max_point)
    expected_body = find_shallowest_rigid_body(str(usd_path), within_default_prim=True, relative_to_default_prim=True)
    assert geometry.get_contact_body_path() == expected_body == "/Group/Body"
    assert usd_path.read_bytes() == original_content
    return True


def test_usd_bounds_and_contact_body_match_main_helpers(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_usd_bounds_and_contact_body_match_main_helpers, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"


def _test_usd_scale_none_keeps_authored_root_scale(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.object_geometry import ObjectGeometry

    usd_path = asset_dir / "root_scaled.usda"
    _write_root_scaled_body_usd(usd_path)

    def size(scale):
        return ObjectGeometry(UsdFileCfg(usd_path=str(usd_path), scale=scale)).get_bounding_box().size[0]

    assert torch.allclose(size(None), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(size((1.0, 1.0, 1.0)), torch.tensor([2.0, 2.0, 2.0]))
    assert ObjectGeometry(UsdFileCfg(usd_path=str(usd_path))).get_contact_body_path() == ""
    return True


def test_usd_scale_none_keeps_authored_root_scale(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_usd_scale_none_keeps_authored_root_scale, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"


def _test_variants_select_bounds_and_contact_body(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd

    from isaaclab_arena.assets.object_geometry import ObjectGeometry

    size_path = asset_dir / "size.usda"
    layout_path = asset_dir / "layout.usda"
    _write_size_variant_usd(size_path)
    _write_layout_variant_usd(layout_path)
    original_content = size_path.read_bytes()

    def size(variants):
        cfg = UsdFileCfg(usd_path=str(size_path), scale=(1.0, 1.0, 1.0), variants=variants)
        return ObjectGeometry(cfg).get_bounding_box().size[0]

    assert torch.allclose(size(None), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(size({"size": "large"}), torch.tensor([4.0, 4.0, 4.0]))
    for selection, body_path in (("first", "/FirstBody"), ("second", "/SecondBody")):
        cfg = UsdFileCfg(usd_path=str(layout_path), variants={"layout": selection})
        assert ObjectGeometry(cfg).get_contact_body_path() == body_path

    assert size_path.read_bytes() == original_content
    stage = Usd.Stage.Open(str(size_path))
    assert stage.GetDefaultPrim().GetVariantSets().GetVariantSet("size").GetVariantSelection() == "small"
    return True


def test_variants_select_bounds_and_contact_body(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_variants_select_bounds_and_contact_body, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"


def _test_contact_body_keeps_shallowest_rigid_body(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object_geometry import ObjectGeometry

    two_bodies = asset_dir / "two_bodies.usda"
    stage = Usd.Stage.CreateNew(str(two_bodies))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Asset/A").GetPrim())
    UsdGeom.Xform.Define(stage, "/Asset/B")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Asset/B/C").GetPrim())
    stage.GetRootLayer().Save()
    assert ObjectGeometry(UsdFileCfg(usd_path=str(two_bodies))).get_contact_body_path() == "/A"

    no_body = asset_dir / "no_body.usda"
    stage = Usd.Stage.CreateNew(str(no_body))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/Cube")
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="No rigid body"):
        ObjectGeometry(UsdFileCfg(usd_path=str(no_body))).get_contact_body_path()
    return True


def test_contact_body_keeps_shallowest_rigid_body(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_contact_body_keeps_shallowest_rigid_body, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"


def _test_usd_without_default_prim_fails_clearly(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.object_geometry import ObjectGeometry

    usd_path = asset_dir / "no_default_prim.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Cube")
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="default prim"):
        ObjectGeometry(UsdFileCfg(usd_path=str(usd_path))).get_bounding_box()
    return True


def test_usd_without_default_prim_fails_clearly(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_usd_without_default_prim_fails_clearly, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"


def _test_primitive_bounds_and_contact_path(simulation_app, shape_name, shape_kwargs, expected_max) -> bool:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_geometry import ObjectGeometry

    shape_cls = getattr(sim_utils, shape_name)
    geometry = ObjectGeometry(shape_cls(rigid_props=sim_utils.RigidBodyPropertiesCfg(), **shape_kwargs))
    bounds = geometry.get_bounding_box()
    assert torch.allclose(bounds.max_point[0], torch.tensor(expected_max))
    assert torch.allclose(bounds.min_point[0], -torch.tensor(expected_max))
    assert geometry.get_contact_body_path() == ""
    return True


@pytest.mark.parametrize(
    "shape_name, shape_kwargs, expected_max",
    [
        ("CuboidCfg", {"size": (1.0, 2.0, 3.0)}, (0.5, 1.0, 1.5)),
        ("SphereCfg", {"radius": 0.1}, (0.1, 0.1, 0.1)),
        ("CylinderCfg", {"radius": 0.1, "height": 0.4, "axis": "X"}, (0.2, 0.1, 0.1)),
        ("ConeCfg", {"radius": 0.1, "height": 0.4}, (0.1, 0.1, 0.2)),
        ("CapsuleCfg", {"radius": 0.1, "height": 0.4, "axis": "Z"}, (0.1, 0.1, 0.3)),
    ],
)
def test_primitive_bounds_and_contact_path(shape_name, shape_kwargs, expected_max):
    result = run_function_with_persistent_simulation_app(
        _test_primitive_bounds_and_contact_path,
        headless=HEADLESS,
        shape_name=shape_name,
        shape_kwargs=shape_kwargs,
        expected_max=expected_max,
    )
    assert result, "Test failed"


def _test_unsupported_geometry_requests_fail(simulation_app) -> bool:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_geometry import ObjectGeometry, get_geometry_key

    with pytest.raises(AssertionError, match="rigid_props"):
        ObjectGeometry(sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0))).get_contact_body_path()
    with pytest.raises(AssertionError, match="bounding_box"):
        ObjectGeometry(sim_utils.DomeLightCfg()).get_bounding_box()
    with pytest.raises(AssertionError, match="one concrete"):
        get_geometry_key(sim_utils.MultiUsdFileCfg(usd_path=["a.usd", "b.usd"]))
    with pytest.raises(AssertionError, match="one concrete"):
        get_geometry_key(sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0))]))
    return True


def test_unsupported_geometry_requests_fail():
    result = run_function_with_persistent_simulation_app(_test_unsupported_geometry_requests_fail, headless=HEADLESS)
    assert result, "Test failed"


def _test_geometry_key_tracks_only_geometric_settings(simulation_app) -> bool:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_geometry import get_geometry_key

    base = sim_utils.UsdFileCfg(usd_path="/tmp/a.usd", scale=(1.0, 2.0, 3.0))
    same_scale_as_list = sim_utils.UsdFileCfg(usd_path="/tmp/a.usd", scale=[1.0, 2.0, 3.0])
    physics_only = sim_utils.UsdFileCfg(
        usd_path="/tmp/a.usd",
        scale=(1.0, 2.0, 3.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        visible=False,
    )
    assert get_geometry_key(same_scale_as_list) == get_geometry_key(base)
    assert get_geometry_key(physics_only) == get_geometry_key(base)
    changed = [
        sim_utils.UsdFileCfg(usd_path="/tmp/b.usd", scale=(1.0, 2.0, 3.0)),
        sim_utils.UsdFileCfg(usd_path="/tmp/a.usd", scale=(1.0, 1.0, 1.0)),
        sim_utils.UsdFileCfg(usd_path="/tmp/a.usd", scale=None),
        sim_utils.UsdFileCfg(usd_path="/tmp/a.usd", scale=(1.0, 2.0, 3.0), variants={"size": "large"}),
    ]
    for cfg in changed:
        assert get_geometry_key(cfg) != get_geometry_key(base)
    small = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0))
    tall = sim_utils.CuboidCfg(size=(1.0, 1.0, 2.0))
    assert get_geometry_key(small) != get_geometry_key(tall)
    return True


def test_geometry_key_tracks_only_geometric_settings():
    result = run_function_with_persistent_simulation_app(
        _test_geometry_key_tracks_only_geometric_settings, headless=HEADLESS
    )
    assert result, "Test failed"
