# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import field

import pytest
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    RunTimeVariationBase,
    VariationBaseCfg,
    VariationBuildContext,
)

HEADLESS = True

TEST_ASSET_NAME = "sphere"
TEST_APPLIED_RADIUS = 0.42
TEST_PER_ENV_VALUE = 0.25
TEST_NUM_ENVS = 3
TEST_SEED = 7


@configclass
class TestBuildTimeVariationCfg(VariationBaseCfg):
    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[TEST_APPLIED_RADIUS],
            high=[TEST_APPLIED_RADIUS],
        ),
    )


class TestBuildTimeVariation(BuildTimeVariationBase):
    __test__ = False

    cfg: TestBuildTimeVariationCfg

    def __init__(self, asset, cfg: TestBuildTimeVariationCfg | None = None, name: str = "test_build_time"):
        super().__init__(cfg=cfg if cfg is not None else TestBuildTimeVariationCfg(), name=name)
        self._asset = asset

    def _realize_at_build_time(self) -> None:
        assert self.sampler is not None
        self._asset.object_cfg.spawn.radius = float(self.sampler.sample(num_samples=1)[0, 0])


class PerEnvContextReadingVariation(BuildTimeVariationBase):
    """Build-time variation that keeps each build context it receives and draws one value per env."""

    __test__ = False

    def __init__(self, name: str = "per_env_context_reading"):
        super().__init__(
            cfg=VariationBaseCfg(
                sampler_cfg=UniformSamplerCfg(low=[TEST_PER_ENV_VALUE], high=[TEST_PER_ENV_VALUE]),
            ),
            name=name,
        )
        self.received_build_contexts: list[VariationBuildContext] = []

    def _realize_at_build_time(self) -> None:
        self.received_build_contexts.append(self.build_context)
        num_envs = self.build_context.num_envs
        self.sampler.sample(num_samples=num_envs, env_ids=torch.arange(num_envs))


def noop_context_reading_event(env, env_ids, asset_cfg):  # noqa: ARG001
    """No-op event term for the run-time context reading variation."""


class RunTimeContextReadingVariation(RunTimeVariationBase):
    """Run-time variation that keeps each build context its build-time preparation receives."""

    __test__ = False

    def __init__(self, name: str = "run_time_context_reading"):
        super().__init__(
            cfg=VariationBaseCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])),
            name=name,
        )
        self.received_build_contexts: list[VariationBuildContext] = []

    def _prepare_at_build_time(self) -> None:
        self.received_build_contexts.append(self.build_context)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_cfg = EventTermCfg(
            func=noop_context_reading_event,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(TEST_ASSET_NAME)},
        )
        return f"{TEST_ASSET_NAME}_{self.name}", event_cfg


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled build-time test variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    assert sphere.name == TEST_ASSET_NAME

    variation = TestBuildTimeVariation(sphere)
    if enabled:
        variation.enable()
    assert variation.enabled is enabled
    sphere.add_variation(variation)

    return IsaacLabArenaEnvironment(
        name="test_build_time_variations",
        scene=Scene(assets=[sphere]),
    )


def _test_disabled_build_time_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    default_radius = sphere.object_cfg.spawn.radius
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert sphere.object_cfg.spawn.radius == default_radius, (
        f"Disabled build-time variation must not mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius'; "
        f"expected {default_radius}, got {sphere.object_cfg.spawn.radius}."
    )
    return True


def _test_enabled_build_time_variation_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    # The radius is stored on SphereCfg as a float32, so compare with a tolerance
    # rather than against the Python double TEST_APPLIED_RADIUS.
    assert sphere.object_cfg.spawn.radius == pytest.approx(TEST_APPLIED_RADIUS, abs=1e-6), (
        f"Enabled build-time variation must mutate '{TEST_ASSET_NAME}.object_cfg.spawn.radius' "
        f"to {TEST_APPLIED_RADIUS}; got {sphere.object_cfg.spawn.radius}."
    )
    return True


def _test_builder_passes_build_context_and_records_per_env_samples(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    per_env_variation = PerEnvContextReadingVariation()
    per_env_variation.enable()
    sphere.add_variation(per_env_variation)
    run_time_context_variation = RunTimeContextReadingVariation()
    run_time_context_variation.enable()
    sphere.add_variation(run_time_context_variation)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(TEST_NUM_ENVS), "--seed", str(TEST_SEED)])

    _, env_kwargs = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert per_env_variation.received_build_contexts == [
        VariationBuildContext(num_envs=TEST_NUM_ENVS, seed=TEST_SEED, variation_key="sphere.per_env_context_reading")
    ]
    # Run-time variations get the context too, for their build-time preparation.
    assert run_time_context_variation.received_build_contexts == [
        VariationBuildContext(num_envs=TEST_NUM_ENVS, seed=TEST_SEED, variation_key="sphere.run_time_context_reading")
    ]
    variation_recorder = env_kwargs["variation_recorder"]
    per_env_record = variation_recorder["sphere.per_env_context_reading"]
    for env_id in range(TEST_NUM_ENVS):
        for episode_idx in (0, 4):
            per_env_value = per_env_record.sample_for_episode(env_id, episode_idx)
            assert per_env_value.tolist() == pytest.approx([TEST_PER_ENV_VALUE], abs=1e-6)
    # Only the build's environments have a per-env value.
    env_outside_build = TEST_NUM_ENVS
    assert per_env_record.sample_for_episode(env_outside_build, 0) is None
    # TestBuildTimeVariation, added by get_test_environment(enabled=True), draws without env_ids,
    # so any env id reports its one shared value.
    shared_record = variation_recorder["sphere.test_build_time"]
    shared_value = shared_record.sample_for_episode(env_outside_build, 4)
    assert shared_value.tolist() == pytest.approx([TEST_APPLIED_RADIUS], abs=1e-6)
    return True


def test_disabled_build_time_variation_not_applied():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_build_time_variation_not_applied,
        headless=HEADLESS,
    )


def test_enabled_build_time_variation_applied():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_build_time_variation_applied,
        headless=HEADLESS,
    )


def test_builder_passes_build_context_and_records_per_env_samples():
    assert run_function_with_persistent_simulation_app(
        _test_builder_passes_build_context_and_records_per_env_samples,
        headless=HEADLESS,
    )
