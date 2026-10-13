# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _condition(env, index=0):
    return env.conditions[:, index]


def _env(values):
    import torch

    return SimpleNamespace(
        num_envs=len(values),
        device="cpu",
        conditions=torch.tensor(values),
        step_dt=0.02,
        episode_length_buf=torch.zeros(len(values), dtype=torch.long),
        extras={},
    )


def _step(tracker, env):
    env.episode_length_buf += 1
    tracker.step(env, env.episode_length_buf)


def _events(tracker, env_id=0):
    return [(event.group_name, event.step) for event in tracker.get_group_events()[env_id]]


def _test_recurring_groups_and_success(_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True], [True]])
    criteria = [CompletionCriteria("success", predicate_sequence=[TrueForConsecutiveStepsCfg(_condition, 3)])]
    baseline = ProgressTracker(criteria, 2, "cpu")
    tracker = ProgressTracker(
        criteria,
        2,
        "cpu",
        tracking_groups=[
            PredicateGroup("instant", [_condition]),
            PredicateGroup("held", [TrueForConsecutiveStepsCfg(_condition, 2)]),
        ],
    )
    for step in range(1, 7):
        _step(tracker, env)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
    assert _events(tracker) == [
        ("instant", 1),
        ("instant", 2),
        ("held", 2),
        ("instant", 3),
        ("instant", 4),
        ("held", 4),
        ("instant", 5),
        ("instant", 6),
        ("held", 6),
    ]
    snapshot = tracker.get_group_events()
    tracker.reset([0])
    env.episode_length_buf[0] = 0
    assert len(snapshot[0]) == 9
    assert tracker.get_group_events()[0] == []
    _step(tracker, env)
    assert _events(tracker, 0) == [("instant", 1)]
    assert _events(tracker, 1)[-1] == ("instant", 7)
    _step(tracker, env)
    assert _events(tracker, 0)[-1] == ("held", 2)
    assert _events(tracker, 1)[-1] == ("held", 8)
    return True


def _test_order_and_fresh_history(_app):
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _env([[False, True]])
    predicates = [partial(_condition, index=0), partial(_condition, index=1)]
    tracker = ProgressTracker(
        [],
        1,
        "cpu",
        tracking_groups=[
            PredicateGroup("ordered", predicates, predicates_are_sequential=True),
            PredicateGroup("unordered", predicates),
        ],
    )
    _step(tracker, env)
    env.conditions[:] = env.conditions.logical_not()
    _step(tracker, env)
    assert _events(tracker) == [("unordered", 2)]
    env.conditions[:] = False
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker)[-1] == ("ordered", 3)
    _step(tracker, env)
    assert len(_events(tracker)) == 2, "A new cycle must observe the first member again."
    env.conditions[:] = True
    _step(tracker, env)
    assert _events(tracker)[-1] == ("unordered", 5)
    _step(tracker, env)
    assert ("ordered", 6) in _events(tracker)
    assert not tracker.is_complete().any(), "Observations alone never make the task succeed."
    return True


def _test_desired_states_and_none_gating(_app):
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = _env([[False, False]])
    predicates = [partial(_condition, index=0), partial(_condition, index=1)]
    tracker = ProgressTracker(
        [],
        1,
        "cpu",
        tracking_groups=[
            PredicateGroup("cleared", [predicates[0]], desired_predicate_state=[False]),
            PredicateGroup("gated", predicates, True, [None, True]),
            PredicateGroup("ignored", predicates, False, [None, True]),
        ],
    )
    _step(tracker, env)
    assert not _events(tracker), "False requires prior satisfaction."
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker) == [("ignored", 2)]
    env.conditions[:] = True
    _step(tracker, env)
    assert ("gated", 3) not in _events(tracker), "The next member starts on the next step."
    env.conditions[:, 0] = False
    _step(tracker, env)
    assert ("cleared", 4) in _events(tracker)
    assert ("gated", 4) in _events(tracker)
    _step(tracker, env)
    assert ("cleared", 5) not in _events(tracker)
    assert ("gated", 5) not in _events(tracker)
    return True


def _test_temporal_group_isolation(_app):
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[False, True], [True, True]])
    counter = TrueForConsecutiveStepsCfg(partial(_condition, index=1), required_steps=2)
    tracker = ProgressTracker(
        [],
        2,
        "cpu",
        env,
        tracking_groups=[
            PredicateGroup("gated", [_condition, counter], True),
            PredicateGroup("independent", [counter]),
        ],
    )
    for _ in range(3):
        _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2)]
    assert _events(tracker, 1) == [("independent", 2), ("gated", 3)]
    # Reusing a configuration must not share counters or count while waiting for the gate.
    env.conditions[0, 0] = True
    _step(tracker, env)
    _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2), ("independent", 4)]
    _step(tracker, env)
    assert _events(tracker, 0)[-2:] == [("gated", 6), ("independent", 6)]
    assert _events(tracker, 1)[-2:] == [("gated", 6), ("independent", 6)]
    _step(tracker, env)
    # Reset one episode with a partially accumulated streak in the other environment.
    tracker.reset([0])
    env.episode_length_buf[0] = 0
    _step(tracker, env)
    assert _events(tracker, 0) == []
    assert _events(tracker, 1)[-1] == ("independent", 8)
    _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2)]
    assert _events(tracker, 1)[-1] == ("gated", 9)
    _step(tracker, env)
    assert _events(tracker, 0)[-1] == ("gated", 3)
    return True


def _test_callable_groups(_app):
    from threading import Lock

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class ReadCondition:
        def __init__(self):
            self.lock = Lock()

        def read(self, env):
            with self.lock:
                return _condition(env)

        def __call__(self, env):
            return self.read(env)

    source = ReadCondition()
    for predicate in (source, source.read, partial(source.read)):
        env = _env([[True]])
        held = TrueForConsecutiveStepsCfg(predicate, required_steps=2)
        criteria = [CompletionCriteria("success", [held])]
        baseline = ProgressTracker(criteria, 1, "cpu")
        tracker = ProgressTracker(
            criteria,
            1,
            "cpu",
            tracking_groups=[
                PredicateGroup("instant", [predicate]),
                PredicateGroup("held", [held]),
                PredicateGroup("ordered", [held, held], predicates_are_sequential=True),
            ],
        )
        for _ in range(5):
            _step(tracker, env)
            baseline.step(env, env.episode_length_buf)
            assert tracker.get_state() == baseline.get_state()
        for name, expected_steps in (("instant", [1, 2, 3, 4, 5]), ("held", [2, 4]), ("ordered", [4])):
            assert [step for group, step in _events(tracker) if group == name] == expected_steps
    return True


def _test_finished_episode_groups(_app, output_dir):
    import json
    import torch

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    def always(env):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def initially_true(env):
        return env.episode_length_buf <= 2

    def finished(env):
        return env.episode_length_buf >= torch.tensor([2, 5], device=env.device)

    class ObservingTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None, success=[CompletionCriteria("finished", [finished])])

        def get_tracking_groups(self):
            held = TrueForConsecutiveStepsCfg(always, required_steps=2)
            return [
                PredicateGroup("instant", [always]),
                PredicateGroup("held", [held]),
                PredicateGroup("ordered", [held, held], predicates_are_sequential=True),
                PredicateGroup(
                    "cleared",
                    [TrueForConsecutiveStepsCfg(initially_true, required_steps=2)],
                    desired_predicate_state=[False],
                ),
            ]

    description = IsaacLabArenaEnvironment(
        name="predicate_group_episode_limit", scene=Scene(), task=ObservingTask(), embodiment=NoEmbodiment()
    )
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False))
    env = builder.make_registered()
    base = env.unwrapped
    results_path = output_dir / "episode_results.jsonl"
    try:
        base.configure_episode_limit(2)
        base.episode_recorder_manager.set_output_path(results_path)
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=base.device)
        finished_events = None
        for step in range(1, 8):
            env.step(actions)
            if step == 2:
                assert base.active_episode_mask.tolist() == [False, True]
                finished_events = base.progress_tracker.get_group_events()[0]
            if step >= 2:
                assert base.progress_tracker.get_group_events()[0] == finished_events
                assert base.extras["progress_tracking"]["group_events"][0] == finished_events
        assert not base.active_episode_mask.any()
        assert _events(base.progress_tracker, 0) == [("instant", 1), ("instant", 2), ("held", 2)]
        assert _events(base.progress_tracker, 1) == [
            ("instant", 1),
            ("instant", 2),
            ("held", 2),
            ("instant", 3),
            ("cleared", 3),
            ("instant", 4),
            ("held", 4),
            ("ordered", 4),
            ("instant", 5),
        ]
        records = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert len(records) == 2
        for record in records:
            events = record["progress"]["group_events"]
            assert [(event["group_name"], event["step"]) for event in events] == _events(
                base.progress_tracker, record["env_id"]
            )
    finally:
        env.close()
    return True


def _test_recording_and_validation(_app):
    import json

    import pytest

    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker, ProgressTrackingRecorderCfg
    from isaaclab_arena.recording.progress_terms import record_progress_results

    env = _env([[True]])
    group = PredicateGroup("observed", [_condition])
    env.progress_tracker = ProgressTracker([], 1, "cpu", tracking_groups=[group])
    tracker = env.progress_tracker
    with pytest.raises(AssertionError, match="step_index"):
        tracker.step(env)
    with pytest.raises(AssertionError, match="unique"):
        ProgressTracker([], 1, "cpu", tracking_groups=[group, group])
    with pytest.raises(AssertionError, match="At least one"):
        PredicateGroup("invalid", [_condition], desired_predicate_state=[None])
    _step(tracker, env)
    with pytest.raises(AssertionError, match="exactly one"):
        tracker.step(env, env.episode_length_buf)
    assert _events(tracker) == [("observed", 1)]
    cfg = ProgressTrackingRecorderCfg()
    cfg.class_type(cfg, env).record_post_step()
    tracker.reset([0])
    result = json.loads(json.dumps(record_progress_results(env, 0)))
    assert result["progress"] == {
        "group_names": ["observed"],
        "group_events": [{"group_name": "observed", "step": 1}],
        "control_step_dt": 0.02,
    }
    return True


def _test_builder_and_composition(_app):
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.predicate_group import PredicateGroup
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask

    class ObservingTask(NoTask):
        def get_tracking_groups(self):
            return [PredicateGroup("observed", [_condition])]

    task = ObservingTask()
    description = IsaacLabArenaEnvironment(
        name="observation_only",
        scene=Scene(),
        task=task,
        embodiment=NoEmbodiment(),
    )
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
    cfg, _ = builder.compose_manager_cfg()
    assert cfg.terminations.to_dict() == {}
    assert cfg.task_progress.success_criteria == []
    assert cfg.task_progress.tracking_groups[0].name == "observed"
    composite = CompositeTaskBase([task, task], episode_length_s=10, subtasks_are_sequential=True)
    assert [group.name for group in composite.get_tracking_groups()] == ["subtask_0/observed", "subtask_1/observed"]
    assert task.get_tracking_groups()[0].name == "observed"
    return True


def test_recurring_groups_and_success():
    assert run_function_with_persistent_simulation_app(_test_recurring_groups_and_success)


def test_order_and_fresh_history():
    assert run_function_with_persistent_simulation_app(_test_order_and_fresh_history)


def test_desired_states_and_none_gating():
    assert run_function_with_persistent_simulation_app(_test_desired_states_and_none_gating)


def test_temporal_group_isolation():
    assert run_function_with_persistent_simulation_app(_test_temporal_group_isolation)


def test_callable_groups():
    assert run_function_with_persistent_simulation_app(_test_callable_groups)


def test_finished_episode_groups(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_finished_episode_groups, output_dir=tmp_path)


def test_recording_and_validation():
    assert run_function_with_persistent_simulation_app(_test_recording_and_validation)


def test_builder_and_composition():
    assert run_function_with_persistent_simulation_app(_test_builder_and_composition)
