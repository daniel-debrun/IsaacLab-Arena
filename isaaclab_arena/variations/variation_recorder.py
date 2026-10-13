# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, get_variation_key

if TYPE_CHECKING:
    from isaaclab_arena.variations.sampler_base import SamplerBase
    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


@dataclass(frozen=True)
class EnvEpisodeKey:
    """Hashable key identifying one env's draws during one episode."""

    env_id: int
    episode_idx: int


class VariationRecord:
    """Per-variation record of the values drawn for it."""

    def __init__(self, name: str, cfg: VariationBaseCfg | None = None) -> None:
        self.name = name
        self.cfg = cfg
        # Run-time draw, one per (env id, episode index).
        self._samples_by_env_episode: dict[EnvEpisodeKey, Any] = {}
        # Build-time draw shared by all envs; applies to every episode of every env.
        self._build_time_sample: Any = None
        self._has_shared_build_time_sample = False
        # Build-time per-env draws; each applies to every episode of its env, so release_episode keeps them.
        self._build_time_samples_by_env: dict[int, Any] = {}

    def record_runtime_sample(self, sample: Any, env_ids: Sequence[int], episode_indices: Sequence[int]) -> None:
        """Record each row of ``sample`` against the (env id, episode index) it was drawn for.

        Each (env id, episode index) is expected to be drawn for at most once.

        Args:
            sample: The drawn sample; row ``i`` is the value for the ``i``-th env in ``env_ids``.
            env_ids: The env ids the sample's rows correspond to.
            episode_indices: The episode index each row was drawn during, aligned with ``env_ids``.
        """
        for row, (env_id, episode_idx) in enumerate(zip(env_ids, episode_indices)):
            key = EnvEpisodeKey(env_id, episode_idx)
            assert (
                key not in self._samples_by_env_episode
            ), f"Variation '{self.name}' already recorded a sample for env {env_id}, episode {episode_idx}."
            self._samples_by_env_episode[key] = sample[row]

    def record_buildtime_sample(self, sample: Any, env_ids: Sequence[int] | None = None) -> None:
        """Record a build-time draw: one row shared by all envs, or one row per env in ``env_ids``.

        A shared draw replaces an earlier shared draw. A variation cannot mix shared and per-env
        build-time draws. A later per-env draw for other envs is accepted; an env that already has a
        value is rejected. A rejected draw records nothing.

        Args:
            sample: The drawn rows; a single row when ``env_ids`` is None, otherwise row ``i``
                belongs to ``env_ids[i]``.
            env_ids: The envs the rows were drawn for, or None for a draw shared by all envs.
        """
        if env_ids is None:
            self._record_shared_build_time_sample(sample)
        else:
            self._record_per_env_build_time_samples(sample, list(env_ids))

    def _record_shared_build_time_sample(self, sample: Any) -> None:
        assert (
            len(sample) == 1
        ), f"Variation '{self.name}' build-time draw expected a single sample for all envs; got {len(sample)}."
        assert (
            not self._build_time_samples_by_env
        ), f"Variation '{self.name}' mixes shared and per-environment build-time draws."
        self._build_time_sample = sample[0]
        self._has_shared_build_time_sample = True

    def _record_per_env_build_time_samples(self, sample: Any, env_ids: list[int]) -> None:
        # Validate the whole draw first, so a rejected draw leaves no partial record.
        assert len(sample) == len(env_ids), (
            f"Variation '{self.name}' per-environment build-time draw has {len(sample)} rows for "
            f"{len(env_ids)} environments."
        )
        assert (
            not self._has_shared_build_time_sample
        ), f"Variation '{self.name}' mixes shared and per-environment build-time draws."
        assert len(set(env_ids)) == len(
            env_ids
        ), f"Variation '{self.name}' per-environment build-time draw repeats an environment id: {env_ids}."
        already_recorded_env_ids = sorted(set(env_ids) & self._build_time_samples_by_env.keys())
        assert (
            not already_recorded_env_ids
        ), f"Variation '{self.name}' already recorded a build-time sample for envs {already_recorded_env_ids}."
        for row, env_id in enumerate(env_ids):
            self._build_time_samples_by_env[env_id] = sample[row]

    def sample_for_episode(self, env_id: int, episode_idx: int) -> Any:
        """Return the value drawn for ``env_id``'s ``episode_idx``, or ``None`` if none was drawn.

        A run-time draw made during that episode comes first, then the env's build-time draw,
        then the build-time draw shared by all envs.
        """
        key = EnvEpisodeKey(env_id, episode_idx)
        if key in self._samples_by_env_episode:
            return self._samples_by_env_episode[key]
        if env_id in self._build_time_samples_by_env:
            return self._build_time_samples_by_env[env_id]
        return self._build_time_sample

    def release_episode(self, env_id: int, episode_idx: int) -> None:
        """Release the run-time sample retained for a finished episode."""
        self._samples_by_env_episode.pop(EnvEpisodeKey(env_id, episode_idx), None)


class VariationRecorder:
    """Records samples drawn by attached variations."""

    def __init__(self) -> None:
        # Records are keyed by: "{asset_name}.{variation_name}"
        self.records: dict[str, VariationRecord] = {}
        self.placement_record: VariationRecord | None = None
        # Bound after env construction; supplies the episode index for per-env run-time draws.
        self._env: Any = None

    def bind_env(self, env: Any) -> None:
        """Bind the env so run-time draws can be attributed to its current episode index."""
        self._env = env

    def __getitem__(self, key: str) -> VariationRecord:
        """Return the record stored under "{asset_name}.{variation_name}"."""
        return self.records[key]

    def __contains__(self, key: str) -> bool:
        """Whether a record is stored under "{asset_name}.{variation_name}"."""
        return key in self.records

    def release_episode(self, env_id: int, episode_idx: int) -> None:
        """Release all run-time variation and placement samples for a finished episode."""
        for record in self.records.values():
            record.release_episode(env_id, episode_idx)
        if self.placement_record is not None:
            self.placement_record.release_episode(env_id, episode_idx)

    def attach(self, variations: dict[str, list[VariationBase]]) -> None:
        """Attach every enabled variation in ``variations`` under "{asset_name}.{variation_name}"."""
        for asset_name, asset_variations in variations.items():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                variation_key = get_variation_key(asset_name, variation.name)
                assert (
                    variation_key not in self.records
                ), f"VariationRecorder: asset_name '{variation_key}' is already attached."

                # Create a record for the variation
                record = VariationRecord(name=variation_key, cfg=variation.cfg)
                self.records[variation_key] = record

                variation.add_sample_listener(
                    self._make_listener(record, build_time=isinstance(variation, BuildTimeVariationBase))
                )

    def attach_placement_sampler(
        self,
        sampler: SamplerBase,
        serializer: Callable[[Any], Any],
    ) -> None:
        """Attach a placement sampler without representing it as a variation."""
        assert self.placement_record is None, "VariationRecorder already has an attached placement sampler."
        self.placement_record = VariationRecord(name="placement")
        sampler.add_listener(self._make_listener(self.placement_record, serializer))

    def _make_listener(
        self,
        record: VariationRecord,
        serializer: Callable[[Any], Any] | None = None,
        *,
        build_time: bool = False,
    ) -> Callable[[Any, torch.Tensor | None], None]:
        """Create a sampler listener using shared episode attribution.

        Args:
            record: The record that receives the samples.
            serializer: Converts each sample before it is recorded, if given.
            build_time: Whether the sampler belongs to a build-time variation. Its per-env draws are then
                kept for every episode of each env instead of the current episode.

        Returns:
            The listener to register with the sampler.
        """

        def on_sample(sample: Any, env_ids: torch.Tensor | None = None) -> None:
            if serializer is not None:
                sample = serializer(sample)
            if isinstance(sample, torch.Tensor):
                sample = sample.detach().cpu()
            if env_ids is None:
                # Shared draw of either lifecycle: applies to every episode of every env.
                record.record_buildtime_sample(sample)
                return
            if build_time:
                # Per-env build-time draw: fixed for every episode of each env. It happens
                # before the env exists, so it needs no bind_env().
                record.record_buildtime_sample(sample, env_ids.tolist())
                return
            assert self._env is not None, "VariationRecorder needs bind_env() before per-env draws."
            env_id_list = env_ids.tolist()
            episode_indices = [self._env.get_episode_index(env_id) for env_id in env_id_list]
            record.record_runtime_sample(sample, env_id_list, episode_indices)

        return on_sample
