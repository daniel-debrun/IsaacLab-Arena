# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_camera"
VARIATION_NAME = f"camera_intrinsics_{CAMERA_NAME}"
EVENT_NAME = f"{CAMERA_NAME}_intrinsics_variation"
# (d_fx, d_fy); distinct signs catch axis/sign swaps.
TEST_DELTAS = (0.1, -0.1)


def get_test_environment(*, camera_intrinsics_variation_enabled: bool):
    """Build a minimal droid-embodiment env with an optional camera intrinsics variation."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS)
    if camera_intrinsics_variation_enabled:
        embodiment.get_variation(VARIATION_NAME).apply_cfg(
            CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(TEST_DELTAS), high=list(TEST_DELTAS)))
        )
        embodiment.get_variation(VARIATION_NAME).enable()

    return IsaacLabArenaEnvironment(name="test_camera_intrinsics_variation", embodiment=embodiment, scene=Scene())


def _test_disabled_camera_intrinsics_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    assert not hasattr(env_cfg.events, EVENT_NAME), (
        f"Disabled variation must not add '{EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_camera_intrinsics_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.camera_intrinsics_variation import apply_camera_intrinsics_from_sampler

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    assert hasattr(
        env_cfg.events, EVENT_NAME
    ), f"Expected env_cfg.events to contain '{EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, EVENT_NAME)
    assert event_cfg.func is apply_camera_intrinsics_from_sampler
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == CAMERA_NAME
    return True


def _test_disabled_camera_intrinsics_variation_keeps_tiled_camera(simulation_app):
    from isaaclab.sensors import TiledCameraCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    for name in arena_env.embodiment.camera_config.camera_names():
        assert getattr(env_cfg.scene, name).renderer_cfg.class_type is not SeparateCameraRenderer
    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, TiledCameraCfg), (
        "With the intrinsics variation disabled, the camera should keep the embodiment's "
        f"tiled default; got {type(camera_cfg).__name__}."
    )
    return True


def _test_enabled_camera_intrinsics_variation_forces_untiled_camera(simulation_app):
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(camera_intrinsics_variation_enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).build_env_cfg()

    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    for name in arena_env.embodiment.camera_config.camera_names():
        camera_cfg = getattr(env_cfg.scene, name)
        assert camera_cfg.renderer_cfg.class_type is SeparateCameraRenderer
        assert not isinstance(camera_cfg, TiledCameraCfg)
    camera_cfg = getattr(env_cfg.scene, CAMERA_NAME)
    assert isinstance(camera_cfg, CameraCfg) and not isinstance(camera_cfg, TiledCameraCfg), (
        "Enabling the intrinsics variation must force the camera untiled so per-env "
        f"perturbations take effect; got {type(camera_cfg).__name__}."
    )
    return True


def _test_untiled_camera_cfg_preserves_source_and_renderer_settings(simulation_app):
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    rig = DroidCameraCfg()
    rig.wrist_camera.renderer_cfg.class_type = IsaacRtxRenderer
    rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel = 4
    rig.wrist_camera.renderer_cfg.enable_scene_partitioning = False
    rig.set_use_tiled_camera(False)
    untiled = rig.get_cfg()
    assert untiled.wrist_camera.renderer_cfg.class_type is SeparateCameraRenderer
    assert untiled.wrist_camera.renderer_cfg.global_settings.samples_per_pixel == 4
    assert not untiled.wrist_camera.renderer_cfg.enable_scene_partitioning
    untiled.wrist_camera.renderer_cfg.global_settings.samples_per_pixel = 8
    assert rig.wrist_camera.renderer_cfg.global_settings.samples_per_pixel == 4
    assert rig.wrist_camera.renderer_cfg.class_type is IsaacRtxRenderer
    rig.set_use_tiled_camera(True)
    tiled = rig.get_cfg()
    assert isinstance(tiled.wrist_camera, TiledCameraCfg)
    assert tiled.wrist_camera.renderer_cfg.class_type is IsaacRtxRenderer
    # A rig declared with TiledCameraCfg fields must also honor the false flag.
    tiled.set_use_tiled_camera(False)
    converted = tiled.get_cfg()
    assert not isinstance(converted.wrist_camera, TiledCameraCfg)
    assert converted.wrist_camera.renderer_cfg.class_type is SeparateCameraRenderer
    # A returned copy can switch back to tiled rendering without keeping the independent renderer.
    converted.set_use_tiled_camera(True)
    restored = converted.get_cfg()
    assert isinstance(restored.wrist_camera, TiledCameraCfg)
    assert restored.wrist_camera.renderer_cfg.class_type is IsaacRtxRenderer
    assert converted.wrist_camera.renderer_cfg.class_type is SeparateCameraRenderer
    return True


def _test_untiled_camera_cfg_rejects_unsupported_settings(simulation_app):
    from isaaclab.renderers import RendererCfg

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    for field, value, message in [
        ("data_types", ["depth"], "only RGB/RGBA"),
        ("isp_cfg", object(), "do not support ISP"),
        ("background_color", (1.0, 0.0, 0.0), "custom background"),
        ("renderer_cfg", RendererCfg(renderer_type="other"), "Isaac RTX"),
    ]:
        rig = DroidCameraCfg()
        rig.set_use_tiled_camera(False)
        setattr(rig.external_camera, field, value)
        with pytest.raises(AssertionError, match=message):
            rig.get_cfg()
    rig = DroidCameraCfg()
    rig.set_use_tiled_camera(False)
    rig.external_camera.renderer_cfg.class_type = object
    with pytest.raises(AssertionError, match="custom renderer"):
        rig.get_cfg()
    return True


def _test_camera_intrinsics_variation_realized_at_runtime(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    env = ArenaEnvBuilder(
        get_test_environment(camera_intrinsics_variation_enabled=True), arena_env_builder_cfg_from_argparse(args_cli)
    ).make_registered()
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    sensor_prim = camera._sensor_prims[0]
    # Read nominal apertures from the rig config so the test tracks its source of truth.
    nominal_spawn = getattr(DroidCameraCfg(), CAMERA_NAME).spawn
    d_fx, d_fy = TEST_DELTAS
    expected_horizontal_aperture = nominal_spawn.horizontal_aperture / (1.0 + d_fx)
    expected_vertical_aperture = nominal_spawn.vertical_aperture / (1.0 + d_fy)

    assert sensor_prim.GetHorizontalApertureAttr().Get() == pytest.approx(expected_horizontal_aperture)
    assert sensor_prim.GetVerticalApertureAttr().Get() == pytest.approx(expected_vertical_aperture)

    env.close()
    return True


def _test_camera_intrinsics_variation_changes_render_independently(simulation_app):
    import numpy as np
    import torch

    import cv2

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer
    from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    arena_env = PickAndPlaceMapleTableEnvironment().build(
        PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True, hdr="home_office_robolab")
    )
    rig = arena_env.embodiment.camera_config
    for name in rig.camera_names():
        camera_cfg = getattr(rig, name)
        camera_cfg.width = 640
        camera_cfg.height = 360
    # Exercise both aliases while the variation activates untiled rendering for the whole rig.
    rig.external_camera_2.data_types = ["rgb", "rgba"]
    variation = arena_env.embodiment.get_variation(VARIATION_NAME)
    variation.apply_cfg(CameraIntrinsicsVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0.0, 0.0], high=[0.0, 0.0])))
    variation.enable()
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, placement_seed=42, resolve_on_reset=False)
    ).make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        for name in rig.camera_names():
            sensor = base.scene[name]
            assert isinstance(sensor._renderer, SeparateCameraRenderer)
            assert len(sensor._render_data.children) == 2
            assert obs["camera_obs"][name + "_rgb"].shape == (2, 360, 640, 3)
        rgba_camera = base.scene["external_camera_2"]
        torch.testing.assert_close(rgba_camera.data.output["rgb"].torch, rgba_camera.data.output["rgba"].torch[..., :3])
        camera = base.scene[CAMERA_NAME]
        term = base.event_manager.get_term_cfg(EVENT_NAME)
        nominal_intrinsics = camera.data.intrinsic_matrices.torch.clone()
        t_W_C = camera.data.pos_w.torch.clone()
        q_W_C = camera.data.quat_w_world.torch.clone()

        def capture():
            for _ in range(12):
                base.sim.render()
                camera.update(base.step_dt, force_recompute=True)
                _ = camera.data.output["rgb"]
            return camera.data.output["rgb"].torch.cpu().numpy().copy()

        def apply(deltas, env_ids):
            sampler = UniformSampler(low=[0.0, 0.0], high=[0.0, 0.0])
            sampler.set_replay_sampler(lambda count, ids: [[deltas[int(i)], deltas[int(i)]] for i in ids])
            term.func(
                base, torch.tensor(env_ids, device=base.device), asset_cfg=term.params["asset_cfg"], sampler=sampler
            )

        def assert_image_scale(reference, current, expected):
            cv2.setRNGSeed(42)
            sift = cv2.SIFT_create(nfeatures=2000)
            keypoints0, descriptors0 = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
            keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), None)
            assert descriptors0 is not None and descriptors1 is not None, "Camera rendered no usable scene features"
            matches = cv2.BFMatcher().knnMatch(descriptors0, descriptors1, k=2)
            good = [a for a, b in matches if a.distance < 0.7 * b.distance]
            assert len(good) >= 20, "Too few matched features to measure rendered zoom"
            pixels0 = np.float32([keypoints0[m.queryIdx].pt for m in good])
            pixels1 = np.float32([keypoints1[m.trainIdx].pt for m in good])
            affine, inliers = cv2.estimateAffinePartial2D(
                pixels0, pixels1, method=cv2.RANSAC, ransacReprojThreshold=2.0
            )
            assert affine is not None and inliers.sum() >= 20
            measured = float(np.linalg.norm(affine[:, 0]))
            assert measured == pytest.approx(expected, abs=0.02), f"Rendered scale {measured}, expected {expected}"

        baseline = capture()
        apply([0.0, 0.5], [0, 1])
        mixed = capture()
        for env_id, scale in enumerate([1.0, 1.5]):
            assert_image_scale(baseline[env_id], mixed[env_id], scale)
        apply([-0.3, 0.5], [0])
        changed = capture()
        for env_id, scale in enumerate([0.7, 1.5]):
            assert_image_scale(baseline[env_id], changed[env_id], scale)
        torch.testing.assert_close(camera.data.pos_w.torch, t_W_C)
        torch.testing.assert_close(camera.data.quat_w_world.torch, q_W_C)
        torch.testing.assert_close(
            camera.data.intrinsic_matrices.torch[:, 0, 0] / nominal_intrinsics[:, 0, 0],
            torch.tensor([0.7, 1.5], device=base.device),
        )
        # A normal reset restores nominal samples and reuses the existing render products.
        products = [child.render_product.path for child in camera._render_data.children]
        env.reset()
        torch.testing.assert_close(camera.data.intrinsic_matrices.torch, nominal_intrinsics)
        assert [child.render_product.path for child in camera._render_data.children] == products
        render_data = [base.scene[name]._render_data for name in rig.camera_names()]
    finally:
        env.close()
    assert all(child.render_product is None for data in render_data for child in data.children)
    return True


def _run_camera_intrinsics_variation_realized_at_runtime(render_check=False):
    """Run the runtime intrinsics check in a disposable Kit process."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.enable_cameras = ENABLE_CAMERAS
    if not HEADLESS:
        args_cli.visualizer = ["kit"]
    with SimulationAppContext(args_cli) as simulation_app_context:
        check = (
            _test_camera_intrinsics_variation_changes_render_independently
            if render_check
            else _test_camera_intrinsics_variation_realized_at_runtime
        )
        assert check(simulation_app_context.app_launcher.app)


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_not_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_camera_intrinsics_variation_not_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_camera_intrinsics_variation_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_disabled_camera_intrinsics_variation_keeps_tiled_camera():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_camera_intrinsics_variation_keeps_tiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_enabled_camera_intrinsics_variation_forces_untiled_camera():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_camera_intrinsics_variation_forces_untiled_camera,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_camera_intrinsics_variation_realized_at_runtime():
    # Lab 3.0 does not fully release the untiled RTX/Replicator resources used by this test
    # when its USD stage is replaced. Keep that state out of the remaining camera tests.
    run_subprocess([TestConstants.python_path, __file__])


@pytest.mark.with_cameras
def test_untiled_camera_cfg_preserves_source_and_renderer_settings():
    assert run_function_with_persistent_simulation_app(
        _test_untiled_camera_cfg_preserves_source_and_renderer_settings,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )


@pytest.mark.with_cameras
def test_untiled_camera_cfg_rejects_unsupported_settings():
    assert run_function_with_persistent_simulation_app(
        _test_untiled_camera_cfg_rejects_unsupported_settings, headless=HEADLESS, enable_cameras=ENABLE_CAMERAS
    )


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_camera_intrinsics_variation_changes_render_independently():
    run_subprocess([
        TestConstants.python_path,
        "-c",
        (
            "from isaaclab_arena.tests.test_camera_intrinsics_variation import "
            "_run_camera_intrinsics_variation_realized_at_runtime; "
            "_run_camera_intrinsics_variation_realized_at_runtime(render_check=True)"
        ),
    ])


if __name__ == "__main__":
    _run_camera_intrinsics_variation_realized_at_runtime()
