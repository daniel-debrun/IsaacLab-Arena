# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base classes.

A :class:`VariationBase` pairs a target asset with a sampler and a hook that
realises one tweak to the scene. Variations attach to any
:class:`~isaaclab_arena.assets.asset.Asset` and start disabled. Concrete
variations subclass one of two flavors:

* :class:`RunTimeVariationBase` — realised via an event term during simulation
  (e.g. per-reset randomization).
* :class:`BuildTimeVariationBase` — drawn once per build and applied to asset configs
  before the env cfg is composed (e.g. picking a dome-light HDR).
"""

from __future__ import annotations

import hashlib
import torch
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset


def get_variation_key(asset_name: str, variation_name: str) -> str:
    """Return the ``"<asset_name>.<variation_name>"`` key that identifies a variation in an environment.

    Args:
        asset_name: Name of the asset the variation is registered on.
        variation_name: Name of the variation on that asset.
    """
    return f"{asset_name}.{variation_name}"


@dataclass(frozen=True)
class VariationBuildContext:
    """Settings of the environment build a variation is configured for."""

    num_envs: int
    """Number of parallel environments in the build."""

    seed: int | None
    """Seed of this environment build, ArenaEnvBuilderCfg.seed.

    With None, make_generator seeds from system entropy, so draws do not repeat.
    """

    variation_key: str
    """Key of the variation in this build, as returned by get_variation_key."""

    def __post_init__(self) -> None:
        assert (
            isinstance(self.num_envs, int) and not isinstance(self.num_envs, bool) and self.num_envs > 0
        ), f"VariationBuildContext.num_envs must be a positive int; got {self.num_envs!r}."
        assert self.seed is None or (
            isinstance(self.seed, int) and not isinstance(self.seed, bool)
        ), f"VariationBuildContext.seed must be an int or None; got {self.seed!r}."
        assert (
            isinstance(self.variation_key, str) and self.variation_key
        ), f"VariationBuildContext.variation_key must be a non-empty string; got {self.variation_key!r}."

    def make_generator(self) -> torch.Generator:
        """Return a CPU generator for this variation's random draws in this build.

        The stream depends only on seed and variation_key: a build with the same seed repeats it,
        other variations get independent streams, and the global torch random state is neither read
        nor advanced. Each call starts the stream again, so make all of the variation's draws in a
        build from one generator. With a None seed, each generator is seeded from system entropy.
        """
        generator = torch.Generator(device="cpu")
        if self.seed is None:
            generator.seed()
            return generator
        seed_material = f"{self.seed}:{self.variation_key}".encode()
        # Python's hash() of a str changes between processes, so use a stable digest; torch generator
        # seeds are 64-bit, so keep its first 8 bytes. Changing this derivation changes every seeded draw.
        generator_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        generator.manual_seed(generator_seed)
        return generator


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances."""

    enabled: bool = False
    """Whether the variation is applied. Opt in via :meth:`VariationBase.enable` or a cfg override."""

    sampler_cfg: SamplerBaseCfg = field(default_factory=SamplerBaseCfg)
    """Declarative sampler driving this variation. Subclasses set a concrete default."""


class VariationBase(ABC):
    """Variation base class.

    This class enforces that the variation has a name, a config, a sampler,
    a way to enable and disable it, and a way to apply a new config. It also gives
    each variation the asset it was added to (``host``) and the settings of the build
    it is configured for (``build_context``).
    """

    cfg: VariationBaseCfg
    """The configclass instance holding this variation's tunable parameters."""

    name: str
    """Identifier under which this variation is registered on its asset."""

    def __init__(self, cfg: VariationBaseCfg, name: str):
        self.name = name
        self._sampler: SamplerBase
        self._sample_listeners: list[Callable[[Any, Any], None]] = []
        self._replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None = None
        self._build_context: VariationBuildContext | None = None
        self._host_ref: weakref.ReferenceType[Asset] | None = None
        self.apply_cfg(cfg)

    @property
    def host(self) -> Asset | None:
        """Asset this variation was added to; None before it is added or after that asset is garbage-collected."""
        if self._host_ref is None:
            return None
        return self._host_ref()

    def set_host(self, asset: Asset) -> None:
        """Set ``asset`` as this variation's host; Asset.add_variation calls this when it adds the variation.

        A variation has one host. Subclasses may override this to reject an unsupported host
        and then call ``super().set_host(asset)``.

        Args:
            asset: Asset the variation is being added to.
        """
        assert self._host_ref is None, (
            f"Variation '{self.name}' already has host asset '{getattr(self.host, 'name', '<garbage-collected>')}'"
            f" and cannot also be added to '{asset.name}'. Create one variation instance per asset."
        )
        # Weak because an asset can be a configclass field (for example JointStateRecorderCfg.object), and
        # configclass validation follows object attributes without cycle detection: a strong
        # asset -> variation -> asset reference would recurse when that config is validated.
        self._host_ref = weakref.ref(asset)

    @property
    def enabled(self) -> bool:
        """Whether this variation is active and should be built into ``events_cfg``."""
        return self.cfg.enabled

    def enable(self) -> None:
        """Mark this variation as active."""
        self.cfg.enabled = True

    def disable(self) -> None:
        """Mark this variation as inactive."""
        self.cfg.enabled = False

    @property
    def sampler(self) -> SamplerBase:
        """The sampler driving this variation."""
        return self._sampler

    def add_sample_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Subscribe ``listener`` (called as ``listener(sample, env_ids)``) to this variation's samples.

        Listeners are stored on the variation, so ``apply_cfg`` re-binds them onto the
        rebuilt sampler and they survive cfg/sampler swaps.
        """
        self._sample_listeners.append(listener)
        self._sampler.add_listener(listener)

    def set_replay_sampler(
        self,
        replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None,
    ) -> None:
        """Route future draws through an optional replay sampler."""
        self._replay_sampler = replay_sampler
        self._sampler.set_replay_sampler(replay_sampler)

    @property
    def build_context(self) -> VariationBuildContext:
        """Build settings from the latest ``configure_at_build_time`` call; kept until the next call replaces them."""
        assert (
            self._build_context is not None
        ), f"Variation '{self.name}' has no build context; pass a VariationBuildContext to configure_at_build_time()."
        return self._build_context

    def _prepare_at_build_time(self) -> None:
        """Configure prerequisites required before environment construction. Default: no-op.

        A run-time variation overrides this when its later event needs a build-time
        precondition (e.g. forcing its camera untiled so per-env edits take effect).
        The build's settings are available through ``build_context``.
        """

    def _realize_at_build_time(self) -> None:
        """Sample and mutate the bound asset config(s) in place during construction. Default: no-op.

        A build-time variation realises its whole effect here; a run-time variation leaves it a no-op.
        The build's settings are available through ``build_context``.
        """

    def configure_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        """Run this variation's build-time preparation and realization, once per env build.

        Args:
            context: Settings of this build, read by the hooks through ``build_context``.
                Without one, a hook that reads ``build_context`` fails.
        """
        # Replace any earlier context, so a build without one never sees a previous build's settings.
        self._build_context = context
        self._prepare_at_build_time()
        self._realize_at_build_time()

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Apply new ``cfg``.

        Replaces ``cfg`` and rebuilds ``sampler`` from ``cfg.sampler_cfg``, re-binding any
        variation-owned sample listeners onto the new sampler. Subclasses with extra derived
        state should override and call ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the ``VariationBaseCfg`` subclass this variation accepts.
        """
        self.cfg = cfg
        assert isinstance(
            cfg.sampler_cfg, SamplerBaseCfg
        ), f"cfg.sampler_cfg must be a SamplerBaseCfg; got {type(cfg.sampler_cfg).__name__}."
        self._sampler = cfg.sampler_cfg.build()
        self._sampler.set_replay_sampler(self._replay_sampler)
        # Re-bind variation-owned listeners so a cfg/sampler swap doesn't drop subscriptions.
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)


class RunTimeVariationBase(VariationBase):
    """Variation realised at run time via an ``EventTermCfg``.

    Use when the underlying property can be flipped during simulation (e.g.
    visual color, initial pose, mass).
    """

    @abstractmethod
    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the ``(name, cfg)`` event term that realises this variation."""
        ...


class BuildTimeVariationBase(VariationBase):
    """Variation drawn once per build and applied before the env is built.

    Use for properties that can't change in-flight: HDR maps, USD swaps,
    spawner params baked into a config. Subclasses realise the effect in
    ``_realize_at_build_time``, on ``host`` (the asset the variation was added to)
    or on assets they keep references to.

    A draw without ``env_ids`` gives one value for all environments. For one value per
    environment, draw ``self.build_context.num_envs`` samples with
    ``env_ids=torch.arange(self.build_context.num_envs)``; each environment's value is recorded
    for every episode of that environment. Recorded variation replay supports only the
    shared form. To make a draw follow the build seed, pass
    ``generator=self.build_context.make_generator()`` to a sampler that accepts one, such as
    ChoiceSampler.
    """

    @abstractmethod
    def _realize_at_build_time(self) -> None:
        """Sample and apply this variation to the target configuration once per env build."""
        ...
