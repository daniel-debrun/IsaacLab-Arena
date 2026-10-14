# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for VariationBase: cfg-driven configuration, sampler wiring, the build context and the host asset.

These tests stay in plain Python (no ``SimulationApp``) because none of them needs an Isaac Sim runtime.
"""

import gc
import hashlib
import torch
import weakref
from dataclasses import FrozenInstanceError, field

import pytest
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    RunTimeVariationBase,
    VariationBaseCfg,
    VariationBuildContext,
    get_variation_key,
)


def _noop_event(env, env_ids, asset_cfg):  # noqa: ARG001
    """No-op event term used to build ``EventTermCfg`` instances in these tests."""


@configclass
class _CustomVariationCfg(VariationBaseCfg):
    """Test-only cfg that adds tunables on top of :class:`VariationBaseCfg`."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0], high=[1.0, 1.0]),
    )

    asset_name: str = "test_asset"
    """Scene-entity name of the asset this variation targets."""

    scale: float = 1.0
    """Arbitrary tunable factor exposed through the cfg."""


_CUSTOM_VARIATION_NAME = "test_custom_variation"


class _CustomVariation(RunTimeVariationBase):
    """Minimal run-time variation used to exercise ``VariationBase`` plumbing."""

    cfg: _CustomVariationCfg

    def __init__(self, cfg: _CustomVariationCfg | None = None, name: str = _CUSTOM_VARIATION_NAME):
        super().__init__(cfg=cfg if cfg is not None else _CustomVariationCfg(), name=name)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (
            f"{self.cfg.asset_name}_{self.name}",
            EventTermCfg(
                func=_noop_event,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(self.cfg.asset_name)},
            ),
        )


def test_default_cfg_populates_variation_state():
    variation = _CustomVariation()

    assert isinstance(variation, RunTimeVariationBase)
    assert variation.name == _CUSTOM_VARIATION_NAME
    assert isinstance(variation.cfg, _CustomVariationCfg)
    assert variation.cfg.asset_name == "test_asset"
    assert variation.cfg.scale == 1.0
    assert variation.enabled is False
    assert isinstance(variation.sampler, UniformSampler)
    assert tuple(variation.sampler.shape_per_sample) == (2,)


def test_custom_cfg_flows_through_to_variation():
    cfg = _CustomVariationCfg(
        enabled=True,
        asset_name="cube",
        scale=2.5,
        sampler_cfg=UniformSamplerCfg(low=[-1.0], high=[1.0]),
    )
    variation = _CustomVariation(cfg)

    assert variation.cfg is cfg
    assert variation.enabled is True
    assert variation.cfg.asset_name == "cube"
    assert variation.cfg.scale == 2.5
    assert tuple(variation.sampler.shape_per_sample) == (1,)


def test_enable_disable_toggles_cfg_flag():
    variation = _CustomVariation()
    assert not variation.enabled

    variation.enable()
    assert variation.enabled and variation.cfg.enabled

    variation.disable()
    assert not variation.enabled and not variation.cfg.enabled


def test_apply_cfg_replaces_cfg_and_rebuilds_sampler():
    variation = _CustomVariation(
        _CustomVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])),
    )
    original_sampler = variation.sampler

    new_cfg = _CustomVariationCfg(
        enabled=True,
        scale=10.0,
        sampler_cfg=UniformSamplerCfg(low=[5.0, 5.0, 5.0], high=[10.0, 10.0, 10.0]),
    )
    variation.apply_cfg(new_cfg)

    assert variation.cfg is new_cfg
    assert variation.enabled
    assert variation.cfg.scale == 10.0
    assert variation.sampler is not original_sampler
    assert tuple(variation.sampler.shape_per_sample) == (3,)
    samples = variation.sampler.sample(num_samples=8)
    assert (samples >= 5.0).all() and (samples <= 10.0).all()


def test_apply_cfg_preserves_replay_sampler():
    variation = _CustomVariation(
        _CustomVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])),
    )
    variation.set_replay_sampler(lambda _count, _env_ids: [[0.25], [0.75]])

    variation.apply_cfg(
        _CustomVariationCfg(sampler_cfg=UniformSamplerCfg(low=[5.0], high=[10.0])),
    )

    assert variation.sampler is not None
    torch.testing.assert_close(variation.sampler.sample(2), torch.tensor([[0.25], [0.75]]))


def test_apply_cfg_with_only_sampler_change_rebuilds_sampler_in_place():
    """A new cfg that differs only in ``sampler_cfg`` should swap the live sampler and leave other state intact."""
    initial_cfg = _CustomVariationCfg(
        enabled=True,
        asset_name="cube",
        scale=3.14,
        sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0]),
    )
    variation = _CustomVariation(initial_cfg)
    original_sampler = variation.sampler

    new_cfg = _CustomVariationCfg(
        enabled=initial_cfg.enabled,
        asset_name=initial_cfg.asset_name,
        scale=initial_cfg.scale,
        sampler_cfg=UniformSamplerCfg(low=[-2.0, -2.0], high=[-1.0, -1.0]),
    )
    variation.apply_cfg(new_cfg)

    assert variation.cfg is new_cfg
    assert variation.enabled is True
    assert variation.cfg.asset_name == "cube"
    assert variation.cfg.scale == 3.14

    assert variation.sampler is not original_sampler
    assert isinstance(variation.sampler, UniformSampler)
    assert tuple(variation.sampler.shape_per_sample) == (2,)
    samples = variation.sampler.sample(num_samples=16)
    assert (samples >= -2.0).all() and (samples <= -1.0).all()


def test_build_event_cfg_uses_configured_asset_name():
    variation = _CustomVariation(_CustomVariationCfg(asset_name="cube"))

    event_name, event_cfg = variation.build_event_cfg()

    assert event_name == f"cube_{_CUSTOM_VARIATION_NAME}"
    assert event_cfg.func is _noop_event
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == "cube"


def test_add_variation_raises_error_if_variation_name_is_already_attached():
    asset = Asset(name="test_asset")
    variation = _CustomVariation()
    asset.add_variation(variation)
    with pytest.raises(AssertionError):
        asset.add_variation(variation)


def test_two_variations_of_same_kind_coexist_when_given_distinct_names():
    """Two instances of the same variation class can attach to one asset when each
    is constructed with a distinct ``name`` override."""
    asset = Asset(name="test_asset")
    variation_a = _CustomVariation(name=f"{_CUSTOM_VARIATION_NAME}_a")
    variation_b = _CustomVariation(name=f"{_CUSTOM_VARIATION_NAME}_b")

    asset.add_variation(variation_a)
    asset.add_variation(variation_b)

    assert asset.get_variation(variation_a.name) is variation_a
    assert asset.get_variation(variation_b.name) is variation_b
    assert set(asset.variations) == {variation_a.name, variation_b.name}


class _BuildContextReadingVariation(BuildTimeVariationBase):
    """Build-time variation whose hooks record the build context they read."""

    def __init__(self, name: str = "build_time"):
        super().__init__(VariationBaseCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])), name=name)
        self.build_steps: list[tuple[str, VariationBuildContext]] = []

    def _prepare_at_build_time(self) -> None:
        self.build_steps.append(("prepare", self.build_context))

    def _realize_at_build_time(self) -> None:
        self.build_steps.append(("realize", self.build_context))


def test_get_variation_key_joins_asset_and_variation_names():
    assert get_variation_key("pickup", "asset_selection") == "pickup.asset_selection"


def test_build_context_reaches_both_hooks():
    variation = _BuildContextReadingVariation()
    context = VariationBuildContext(num_envs=3, seed=17, variation_key="object.build_time")

    variation.configure_at_build_time(context)

    assert variation.build_steps == [("prepare", context), ("realize", context)]
    assert variation.build_context is context


def test_build_context_is_immutable():
    context = VariationBuildContext(num_envs=3, seed=17, variation_key="object.build_time")

    with pytest.raises(FrozenInstanceError):
        context.num_envs = 4


def test_reading_build_context_before_configure_at_build_time_fails():
    variation = _BuildContextReadingVariation()

    with pytest.raises(AssertionError, match="has no build context"):
        variation.build_context


_VALID_BUILD_CONTEXT_FIELDS = {"num_envs": 2, "seed": 1, "variation_key": "asset.variation"}


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("num_envs", 0),
        ("num_envs", True),
        ("num_envs", 2.0),
        ("seed", True),
        ("seed", "1"),
        ("variation_key", ""),
        ("variation_key", None),
    ],
)
def test_build_context_rejects_invalid_field(field_name, invalid_value):
    with pytest.raises(AssertionError, match=f"VariationBuildContext.{field_name} must"):
        VariationBuildContext(**{**_VALID_BUILD_CONTEXT_FIELDS, field_name: invalid_value})


def test_build_context_requires_every_field():
    with pytest.raises(TypeError):
        VariationBuildContext(num_envs=2, seed=1)


def _draw(generator: torch.Generator) -> torch.Tensor:
    """Draw a fixed-size integer sample used to compare generator streams."""
    return torch.randint(0, 1_000_000, (16,), generator=generator)


def test_build_context_generator_is_keyed_by_seed_and_variation_key():
    context = VariationBuildContext(num_envs=1, seed=7, variation_key="pickup.asset_selection")
    original_global_state = torch.random.get_rng_state()

    first_draw = _draw(context.make_generator())
    repeated_draw = _draw(context.make_generator())
    other_num_envs_draw = _draw(
        VariationBuildContext(num_envs=4, seed=7, variation_key="pickup.asset_selection").make_generator()
    )
    other_key_draw = _draw(VariationBuildContext(num_envs=1, seed=7, variation_key="pickup.other").make_generator())
    other_seed_draw = _draw(
        VariationBuildContext(num_envs=1, seed=8, variation_key="pickup.asset_selection").make_generator()
    )

    assert torch.equal(repeated_draw, first_draw)
    assert torch.equal(other_num_envs_draw, first_draw)
    assert not torch.equal(other_key_draw, first_draw)
    assert not torch.equal(other_seed_draw, first_draw)
    assert torch.equal(torch.random.get_rng_state(), original_global_state)


def test_build_context_generator_seed_derivation_is_pinned():
    """Pins how the generator seed is derived, so a change to it, which changes every seeded draw, is deliberate."""
    context = VariationBuildContext(num_envs=1, seed=7, variation_key="pickup.asset_selection")
    expected_generator_seed = int.from_bytes(hashlib.sha256(b"7:pickup.asset_selection").digest()[:8], "big")

    assert torch.equal(_draw(context.make_generator()), _draw(torch.Generator().manual_seed(expected_generator_seed)))


def test_build_context_generator_treats_seed_zero_as_a_seed():
    context = VariationBuildContext(num_envs=1, seed=0, variation_key="asset.variation")

    assert torch.equal(_draw(context.make_generator()), _draw(context.make_generator()))


def test_build_context_generator_without_seed_uses_system_entropy():
    context = VariationBuildContext(num_envs=1, seed=None, variation_key="asset.variation")
    original_global_state = torch.random.get_rng_state()

    first_generator = context.make_generator()
    second_generator = context.make_generator()

    assert first_generator.initial_seed() != second_generator.initial_seed()
    assert torch.equal(torch.random.get_rng_state(), original_global_state)


def test_add_variation_sets_the_host():
    asset = Asset(name="test_asset")
    variation = _CustomVariation()
    assert variation.host is None

    asset.add_variation(variation)

    assert variation.host is asset


def test_variation_can_only_have_one_host():
    asset = Asset(name="first")
    other_asset = Asset(name="second")
    variation = _CustomVariation()
    asset.add_variation(variation)

    with pytest.raises(AssertionError, match="already has host asset 'first' and cannot also be added to 'second'"):
        other_asset.add_variation(variation)

    assert other_asset.get_variations() == []
    assert variation.host is asset


def test_host_survives_apply_cfg():
    asset = Asset(name="test_asset")
    variation = _CustomVariation()
    asset.add_variation(variation)

    variation.apply_cfg(_CustomVariationCfg(enabled=True))

    assert variation.host is asset


def test_host_reference_does_not_keep_the_asset_alive():
    asset = Asset(name="test_asset")
    variation = _CustomVariation()
    asset.add_variation(variation)
    asset_reference = weakref.ref(asset)

    del asset
    gc.collect()

    assert asset_reference() is None
    assert variation.host is None


def test_variation_with_a_host_does_not_create_a_config_validation_cycle():
    # An asset stored in a configclass field is validated with it, and validation follows object
    # attributes without cycle detection, so a strong asset -> variation -> asset reference would
    # recurse until RecursionError.
    asset = Asset(name="test_asset")
    asset.add_variation(_CustomVariation())

    @configclass
    class _AssetHolderCfg:
        target: Asset = asset

    _AssetHolderCfg().validate()
