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
        # Draw made without env ids, by a variation of either lifecycle; applies to every episode of every env.
        self._shared_sample: Any = None
        self._has_shared_sample = False
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

    def record_shared_sample(self, sample: Any) -> None:
        """Record a draw shared by all envs; it applies to every episode of every env.

        A later shared draw replaces it. A shared draw after a per-env build-time draw is rejected.

        Args:
            sample: The drawn sample, holding a single row.
        """
        assert len(sample) == 1, (
            f"Variation '{self.name}' drew {len(sample)} rows without env_ids. A draw shared by all envs has one"
            " row; to draw one row per env, pass env_ids to the sampler."
        )
        assert (
            not self._build_time_samples_by_env
        ), f"Variation '{self.name}' mixes shared and per-environment build-time draws."
        self._shared_sample = sample[0]
        self._has_shared_sample = True

    def record_per_env_build_time_samples(self, sample: Any, env_ids: Sequence[int]) -> None:
        """Record a build-time variation's per-env draw; row ``i`` applies to every episode of ``env_ids[i]``.

        A later draw for other envs is accepted. A draw for an env that already has a value, or for a
        variation with a shared draw, is rejected and records nothing.

        Args:
            sample: The drawn rows, one for each entry of ``env_ids``.
            env_ids: The envs the rows were drawn for.
        """
        env_ids = list(env_ids)
        # Validate the whole draw first, so a rejected draw leaves no partial record.
        assert len(sample) == len(env_ids), (
            f"Variation '{self.name}' per-environment build-time draw has {len(sample)} rows for "
            f"{len(env_ids)} environments."
        )
        assert (
            not self._has_shared_sample
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

        This is the run-time draw made during that episode, else the env's per-env build-time draw,
        else the shared draw. A variation fills either the per-episode or the per-env build-time
        draws, never both.
        """
        key = EnvEpisodeKey(env_id, episode_idx)
        if key in self._samples_by_env_episode:
            return self._samples_by_env_episode[key]
        if env_id in self._build_time_samples_by_env:
            return self._build_time_samples_by_env[env_id]
        return self._shared_sample

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
                    self._make_listener(record, is_build_time_variation=isinstance(variation, BuildTimeVariationBase))
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
        is_build_time_variation: bool = False,
    ) -> Callable[[Any, torch.Tensor | None], None]:
        """Return a listener that stores each draw of a sampler in ``record``.

        A draw without env ids is stored as shared by all envs. A per-env draw is stored for each env's
        current episode, or for every episode of the env when ``is_build_time_variation`` is set.

        Args:
            record: The record that receives the samples.
            serializer: Converts each sample before it is recorded, if given.
            is_build_time_variation: Whether the sampler belongs to a BuildTimeVariationBase.

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
                record.record_shared_sample(sample)
            elif is_build_time_variation:
                # Per-env build-time draw: fixed for every episode of each env. It happens
                # before the env exists, so it needs no bind_env().
                record.record_per_env_build_time_samples(sample, env_ids.tolist())
            else:
                assert self._env is not None, "VariationRecorder needs bind_env() before per-env run-time draws."
                env_id_list = env_ids.tolist()
                episode_indices = [self._env.get_episode_index(env_id) for env_id in env_id_list]
                record.record_runtime_sample(sample, env_id_list, episode_indices)

        return on_sample
