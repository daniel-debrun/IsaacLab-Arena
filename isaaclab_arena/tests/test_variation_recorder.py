# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sample listeners and ``VariationRecorder``.

Listener and recorder mechanics are plain Python (no SimulationApp). The end-to-end test that
records a real HDR draw runs inside ``run_function_with_persistent_simulation_app`` because constructing a dome
light via the registry pulls in ``isaaclab.sim``.
"""

import torch
from dataclasses import field

import pytest
from isaaclab.utils.configclass import configclass

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    RunTimeVariationBase,
    VariationBaseCfg,
    VariationBuildContext,
)
from isaaclab_arena.variations.variation_recorder import VariationRecord, VariationRecorder

HEADLESS = True


@configclass
class _RecorderTestVariationCfg(VariationBaseCfg):
    """Variation cfg whose sampler draws a single scalar."""

    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[1.0]),
    )


class _RecorderTestVariation(BuildTimeVariationBase):
    """Minimal build-time variation that just draws a sample when applied."""

    __test__ = False

    cfg: _RecorderTestVariationCfg

    def __init__(self, cfg: _RecorderTestVariationCfg | None = None, name: str = "recorder_test"):
        super().__init__(cfg=cfg if cfg is not None else _RecorderTestVariationCfg(), name=name)

    def _realize_at_build_time(self) -> None:
        assert self.sampler is not None
        self.sampler.sample(num_samples=1)


class _PerEnvRecorderTestVariation(BuildTimeVariationBase):
    """Build-time variation that draws one value per environment of the build."""

    __test__ = False

    def __init__(self, name: str = "per_env_recorder_test"):
        super().__init__(cfg=_RecorderTestVariationCfg(), name=name)

    def _realize_at_build_time(self) -> None:
        num_envs = self.build_context.num_envs
        self.sampler.sample(num_samples=num_envs, env_ids=torch.arange(num_envs))


class _RecorderTestRuntimeVariation(RunTimeVariationBase):
    """Run-time variation whose samples the tests draw directly from its sampler."""

    __test__ = False

    def __init__(self, name: str = "runtime_recorder_test"):
        super().__init__(cfg=_RecorderTestVariationCfg(), name=name)

    def build_event_cfg(self):
        raise NotImplementedError("Recorder tests draw from the sampler directly.")


@configclass
class _ChoiceRecorderTestVariationCfg(VariationBaseCfg):
    """Variation cfg whose sampler draws items from a per-call list of choices."""

    __test__ = False

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)


class _PerEnvChoiceRecorderTestVariation(BuildTimeVariationBase):
    """Build-time variation that picks one name per environment from a seeded generator."""

    __test__ = False

    choices = ["apple", "banana", "cherry"]

    def __init__(self, name: str = "per_env_choice_recorder_test"):
        super().__init__(cfg=_ChoiceRecorderTestVariationCfg(), name=name)

    def _realize_at_build_time(self) -> None:
        num_envs = self.build_context.num_envs
        self.sampler.sample(
            num_envs,
            choices=self.choices,
            env_ids=torch.arange(num_envs),
            generator=self.build_context.make_generator(),
        )


def test_uniform_sampler_notifies_listeners():
    sampler = UniformSamplerCfg(low=[0.0], high=[1.0]).build()
    seen: list = []
    sampler.add_listener(lambda s, env_ids: seen.append((s, env_ids)))
    result = sampler.sample(num_samples=4, env_ids=[0, 1, 2, 3])
    assert len(seen) == 1
    sample, env_ids = seen[0]
    assert sample is result
    assert tuple(sample.shape) == (4, 1)
    # env_ids are forwarded so listeners can attribute each row to its env.
    assert env_ids == [0, 1, 2, 3]


def test_choice_sampler_notifies_listeners():
    sampler = ChoiceSampler()
    seen: list = []
    sampler.add_listener(lambda s, env_ids: seen.append((s, env_ids)))
    sampler.sample(num_samples=3, choices=["x", "y", "z"], env_ids=[4, 5, 6])
    assert len(seen) == 1
    sample, env_ids = seen[0]
    assert isinstance(sample, list) and len(sample) == 3
    assert env_ids == [4, 5, 6]


def test_variation_listener_survives_sampler_swap():
    """A listener added via the variation must re-bind to the sampler rebuilt by apply_cfg."""
    variation = _RecorderTestVariation()
    seen: list = []
    variation.add_sample_listener(lambda s, _env_ids: seen.append(s))

    # Swap in a new cfg (rebuilds the underlying sampler), then draw a sample.
    variation.apply_cfg(_RecorderTestVariationCfg(sampler_cfg=UniformSamplerCfg(low=[2.0], high=[2.0])))
    sample = variation.sampler.sample(num_samples=1)
    assert len(seen) == 1
    assert seen[0] is sample
    assert sample.item() == pytest.approx(2.0, abs=1e-6)


def test_recorder_records_build_time_sample_for_every_episode():
    """A build-time draw (env_ids=None) is recorded and surfaced for every env/episode."""
    variation = _RecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})

    record = recorder["asset.recorder_test"]
    assert record.sample_for_episode(0, 0) is None

    variation.configure_at_build_time()

    value = record.sample_for_episode(0, 0)
    # Tensor samples are stored detached on CPU.
    assert value.device.type == "cpu"
    # A build-time draw applies to every env and episode, not just (0, 0).
    assert record.sample_for_episode(7, 3).tolist() == value.tolist()
    recorder.release_episode(7, 3)
    assert record.sample_for_episode(7, 3).tolist() == value.tolist()


def test_variation_record_tracks_per_env_episode_values():
    """Each (env, episode) draw is stored and surfaced separately across envs and episodes."""
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg())

    # Runtime-style draw: rows map to the given env ids, all in episode 0.
    record.record_runtime_sample(torch.tensor([[1.0], [2.0]]), env_ids=[2, 5], episode_indices=[0, 0])
    assert record.sample_for_episode(2, 0).tolist() == [1.0]
    assert record.sample_for_episode(5, 0).tolist() == [2.0]
    # An env/episode not drawn for has no recorded value.
    assert record.sample_for_episode(0, 0) is None
    assert record.sample_for_episode(2, 1) is None

    # A draw for env 2 in the next episode lands under its own key.
    record.record_runtime_sample(torch.tensor([[4.0]]), env_ids=[2], episode_indices=[1])
    assert record.sample_for_episode(2, 1).tolist() == [4.0]
    assert record.sample_for_episode(2, 0).tolist() == [1.0]

    # Each (env, episode) is expected to be drawn for at most once.
    with pytest.raises(AssertionError):
        record.record_runtime_sample(torch.tensor([[3.0]]), env_ids=[2], episode_indices=[0])
    record.release_episode(2, 0)
    assert record.sample_for_episode(2, 0) is None
    assert record.sample_for_episode(5, 0).tolist() == [2.0]


class _FakeEnv:
    """Minimal stand-in exposing the attributes ``record_variation_samples`` reads."""

    def __init__(self, recorder: VariationRecorder, episode_index: int = 0) -> None:
        self.variation_recorder = recorder
        self._episode_index = episode_index

    def get_episode_index(self, env_id: int) -> int:  # noqa: ARG002
        return self._episode_index


def test_record_variation_samples_emits_the_per_episode_draw():
    """The episode term emits the draw made for the finishing episode, one value per variation."""
    from isaaclab_arena.recording.common_terms import record_variation_samples

    variation = _RecorderTestRuntimeVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    env = _FakeEnv(recorder, episode_index=0)
    recorder.bind_env(env)

    # One run-time draw for env 0 during episode 0.
    variation.sampler.sample(num_samples=1, env_ids=torch.tensor([0]))

    fields = record_variation_samples(env, env_id=0)
    sample = fields["variations"]["asset.runtime_recorder_test"]
    # The single (1,)-shaped draw is recorded as a flat list, not a list of draws.
    assert isinstance(sample, list) and len(sample) == 1

    recorder.release_episode(env_id=0, episode_idx=0)
    assert recorder["asset.runtime_recorder_test"].sample_for_episode(0, 0) is None

    # A different env in the same episode has nothing recorded, so no variations field is emitted.
    assert record_variation_samples(env, env_id=1) == {}


def test_record_placement_sample_uses_sampler_listener_without_variation():
    """Placement shares episode attribution but is emitted outside ``variations``."""
    from isaaclab_arena.recording.common_terms import record_placement_sample

    variation = _RecorderTestVariation()
    recorder = VariationRecorder()
    recorder.attach_placement_sampler(
        variation.sampler,
        serializer=lambda sample: [{"pose": row.tolist()} for row in sample],
    )
    env = _FakeEnv(recorder, episode_index=2)
    recorder.bind_env(env)

    variation.sampler.sample(num_samples=1, env_ids=torch.tensor([3]))

    fields = record_placement_sample(env, env_id=3)
    assert fields.keys() == {"placement"}
    assert fields["placement"].keys() == {"pose"}
    assert len(fields["placement"]["pose"]) == 1
    recorder.release_episode(env_id=3, episode_idx=2)
    assert recorder.placement_record is not None
    assert recorder.placement_record.sample_for_episode(3, 2) is None


def test_recorder_skips_disabled_variations():
    variation = _RecorderTestVariation()  # disabled by default
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    assert "asset.recorder_test" not in recorder
    assert recorder.records == {}


def test_recorder_duplicate_asset_name_asserts():
    variation = _RecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    with pytest.raises(AssertionError):
        recorder.attach({"asset": [variation]})


def test_recorder_keeps_per_env_build_time_samples_for_every_episode():
    """A per-env build-time draw is recorded before bind_env and applies to every episode of its env."""
    from isaaclab_arena.recording.common_terms import record_variation_samples

    variation = _PerEnvRecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    drawn_samples = []
    variation.add_sample_listener(lambda sample, _env_ids: drawn_samples.append(sample))

    # No bind_env(): build-time draws happen before the env exists.
    variation.configure_at_build_time(
        VariationBuildContext(num_envs=2, seed=3, variation_key="asset.per_env_recorder_test")
    )

    (drawn,) = drawn_samples
    record = recorder["asset.per_env_recorder_test"]
    for env_id in (0, 1):
        for episode_idx in (0, 5):
            value = record.sample_for_episode(env_id, episode_idx)
            assert value.device.type == "cpu"
            assert value.tolist() == drawn[env_id].tolist()
    assert record.sample_for_episode(2, 0) is None

    # Releasing a finished episode drops only run-time samples; the env keeps its build-time value.
    recorder.release_episode(env_id=1, episode_idx=0)
    assert record.sample_for_episode(1, 1).tolist() == drawn[1].tolist()

    env = _FakeEnv(recorder, episode_index=4)
    recorder.bind_env(env)
    assert record_variation_samples(env, env_id=1) == {"variations": {"asset.per_env_recorder_test": drawn[1].tolist()}}


def test_recorder_keeps_per_env_build_time_choices_as_plain_values():
    """A per-env ChoiceSampler draw is recorded per env and emitted as the chosen item, not a tensor."""
    from isaaclab_arena.recording.common_terms import record_variation_samples

    variation = _PerEnvChoiceRecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    drawn_samples = []
    variation.add_sample_listener(lambda sample, _env_ids: drawn_samples.append(sample))

    variation.configure_at_build_time(
        VariationBuildContext(num_envs=3, seed=5, variation_key="asset.per_env_choice_recorder_test")
    )

    (drawn_names,) = drawn_samples
    assert len(drawn_names) == 3
    record = recorder["asset.per_env_choice_recorder_test"]
    env = _FakeEnv(recorder, episode_index=2)
    recorder.bind_env(env)
    for env_id, drawn_name in enumerate(drawn_names):
        assert drawn_name in _PerEnvChoiceRecorderTestVariation.choices
        assert record.sample_for_episode(env_id, 0) == drawn_name
        assert record.sample_for_episode(env_id, 2) == drawn_name
        assert record_variation_samples(env, env_id=env_id) == {
            "variations": {"asset.per_env_choice_recorder_test": drawn_name}
        }


def test_runtime_variation_routing_is_unchanged():
    """Run-time variations: shared draws are shared, per-env draws need bind_env and belong to one episode."""
    variation = _RecorderTestRuntimeVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    record = recorder["asset.runtime_recorder_test"]

    # A run-time variation may draw a shared value during the build (e.g. in _prepare_at_build_time).
    shared_draw = variation.sampler.sample(num_samples=1)
    assert record.sample_for_episode(4, 2).tolist() == shared_draw[0].tolist()

    with pytest.raises(AssertionError, match="needs bind_env"):
        variation.sampler.sample(num_samples=1, env_ids=torch.tensor([0]))

    recorder.bind_env(_FakeEnv(recorder, episode_index=3))
    episode_draw = variation.sampler.sample(num_samples=1, env_ids=torch.tensor([0]))
    assert record.sample_for_episode(0, 3).tolist() == episode_draw[0].tolist()
    # Other episodes of env 0 still report the shared draw.
    assert record.sample_for_episode(0, 2).tolist() == shared_draw[0].tolist()


def test_build_time_record_rejects_invalid_per_env_draws_without_partial_writes():
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg())
    record.record_buildtime_sample(torch.tensor([[1.0], [2.0]]), env_ids=[0, 2])

    with pytest.raises(AssertionError, match="already recorded a build-time sample"):
        record.record_buildtime_sample(torch.tensor([[3.0], [4.0]]), env_ids=[1, 2])
    # The rejected draw wrote nothing: env 1 is still unrecorded and env 2 keeps its value.
    assert record.sample_for_episode(1, 0) is None
    assert record.sample_for_episode(2, 0).tolist() == [2.0]

    with pytest.raises(AssertionError, match="repeats an environment id"):
        record.record_buildtime_sample(torch.tensor([[5.0], [6.0]]), env_ids=[3, 3])
    assert record.sample_for_episode(3, 0) is None

    with pytest.raises(AssertionError, match="rows for"):
        record.record_buildtime_sample(torch.tensor([[7.0]]), env_ids=[4, 5])
    assert record.sample_for_episode(4, 0) is None

    with pytest.raises(AssertionError, match="mixes shared and per-environment"):
        record.record_buildtime_sample(torch.tensor([[8.0]]))
    assert record.sample_for_episode(6, 0) is None

    # A later per-env draw for envs without a value is accepted and leaves earlier values in place.
    record.record_buildtime_sample(torch.tensor([[9.0]]), env_ids=[1])
    assert record.sample_for_episode(1, 0).tolist() == [9.0]
    assert record.sample_for_episode(2, 0).tolist() == [2.0]


def test_build_time_record_rejects_per_env_draw_after_shared_draw():
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg())
    record.record_buildtime_sample(torch.tensor([[1.0]]))

    with pytest.raises(AssertionError, match="mixes shared and per-environment"):
        record.record_buildtime_sample(torch.tensor([[2.0]]), env_ids=[0])
    assert record.sample_for_episode(0, 0).tolist() == [1.0]

    # A repeated shared draw replaces the earlier one, as on main.
    record.record_buildtime_sample(torch.tensor([[3.0]]))
    assert record.sample_for_episode(0, 0).tolist() == [3.0]


def _test_hdr_variation_recorder_captures_chosen_hdr_name(simulation_app):
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.hdr_image_variation import HDRImageVariationCfg

    pool = ["home_office_robolab", "empty_warehouse_robolab", "billiard_hall_robolab"]

    dome_light = AssetRegistry().get_asset_by_name("light")()
    assert isinstance(dome_light, DomeLight)
    variation = dome_light.get_variation("hdr_image")
    variation.apply_cfg(HDRImageVariationCfg(enabled=True, hdr_names=pool))
    scene = Scene(assets=[dome_light])

    # Attach the recorder after the cfg is finalized but before the build, mirroring
    # the order ArenaEnvBuilder.compose_manager_cfg uses.
    recorder = VariationRecorder()
    recorder.attach(scene.get_asset_variations())

    record = recorder["light.hdr_image"]
    assert record.sample_for_episode(0, 0) is None

    variation.configure_at_build_time()

    # The HDR draw is build-time (env_ids=None), so it applies to every env/episode.
    value = record.sample_for_episode(0, 0)
    # The recorder must capture the chosen HDR *name*, not an index.
    assert value in pool
    return True


def test_hdr_variation_recorder_captures_chosen_hdr_name():
    assert run_function_with_persistent_simulation_app(
        _test_hdr_variation_recorder_captures_chosen_hdr_name,
        headless=HEADLESS,
    )
