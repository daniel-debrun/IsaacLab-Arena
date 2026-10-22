# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare replayable settled layouts in an environment separate from evaluation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.resource_cleanup import close_environment


def prepare_settled_placement(cfg: ArenaRunCfg, num_episodes: int, output_dir: Path, rebuild_index: int) -> ArenaRunCfg:
    """Prepare accepted layouts and return an isolated run config that replays them.

    Args:
        cfg: Rebuild configuration with settled_placement enabled.
        num_episodes: Number of accepted layouts required by this rebuild.
        output_dir: Run output directory for accepted layouts and rejection diagnostics.
        rebuild_index: Index used to distinguish preparation artifacts.

    Returns:
        Evaluation configuration referencing accepted layouts, with preparation disabled.
    """
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.scripts.record_placement_layouts import record_placements_to_jsonl
    from isaaclab_arena.variations.variations_hydra import apply_overrides

    preparation = cfg.settled_placement
    assert preparation is not None
    assert num_episodes > 0, "Preparation requires an episode budget"
    assert (
        preparation.max_batches * cfg.environment_builder.num_envs >= num_episodes
    ), "The preparation batch budget cannot supply the requested episode count"
    for check in ("physics_settled", "pose_shift", "articulation_link_shift"):
        assert preparation.settle.validators.get(check, {}).get(
            "enabled", False
        ), f"Automatic preparation requires the '{check}' validator"
    output_dir.mkdir(parents=True, exist_ok=True)
    layouts_path = output_dir / f"settled_placements_rebuild{rebuild_index}.jsonl"
    summary_path = output_dir / f"settled_placements_rebuild{rebuild_index}_summary.json"
    assert not layouts_path.exists() and not summary_path.exists(), "Preparation artifacts already exist"
    sampling_cfg = deepcopy(cfg)
    sampling_cfg.settled_placement = None
    sampling_cfg.environment_builder.resolve_on_reset = True
    sampling_cfg.environment_builder.record_trajectories = False
    builder = build_arena_builder_from_run_cfg(sampling_cfg)
    configure_preparation_resets(builder)
    assert builder.arena_env.env_cfg_callback is None, (
        "Automatic settled placement does not support environment configuration callbacks; "
        "record and validate placements explicitly for custom reset behavior"
    )
    # Placement recordings do not contain variation draws. Resampling could invalidate their geometry or physics.
    configured_variations = builder.get_all_variations()
    if builder.hydra_overrides:
        apply_overrides(configured_variations, builder.hydra_overrides)
    for variations in configured_variations.values():
        assert not any(
            variation.enabled for variation in variations
        ), "Automatic settled placement requires variations to be disabled"
    env = None
    try:
        env_cfg, env_kwargs = builder.build_env_cfg()
        env_cfg.recorders = None
        env = builder.make_registered(env_cfg=env_cfg, env_kwargs=env_kwargs)
        assets = builder.arena_env.get_placement_assets()
        for asset in assets:
            if asset.tags and "embodiment" in asset.tags:
                for key in asset.get_scene_root_keys():
                    assert env.unwrapped.scene[
                        key
                    ].is_fixed_base, "Automatic settled placement currently requires a fixed-base robot"
        params = deepcopy(preparation.settle)
        # Root replay cannot preserve robot joint drift; check embodiment links as well as task articulations.
        params.include_embodiment_links = True
        summary = record_placements_to_jsonl(
            env,
            layouts_path,
            min_layouts=num_episodes,
            max_batches=preparation.max_batches,
            params=params,
            scene_assets=assets,
        )
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "accepted": summary.accepted,
                    "attempted": summary.attempted,
                    "required": num_episodes,
                    "required_settled_steps": params.required_settled_steps,
                    "rejections": [
                        {"env_id": env_id, "batch_index": batch, "reason": reason}
                        for (env_id, batch), reason in summary.rejections.items()
                    ],
                },
                stream,
                indent=2,
            )
        assert (
            summary.accepted == num_episodes
        ), f"Only {summary.accepted}/{num_episodes} settled layouts accepted; see {summary_path}"
    finally:
        close_environment(env)
    replay_cfg = deepcopy(cfg)
    replay_cfg.settled_placement = None
    replay_cfg.environment_builder.recorded_variation_samples_path = str(layouts_path.resolve())
    replay_cfg.environment_builder.resolve_on_reset = True
    return replay_cfg


def configure_preparation_resets(builder) -> None:
    """Use repeatable robot joint resets in both preparation and evaluation.

    Root recordings cannot preserve robot joint randomization. The standard Franka/DROID
    Gaussian reset is made deterministic; other custom robot or task events are unsupported.

    Args:
        builder: Environment builder whose description will be compiled.
    """
    from dataclasses import fields

    arena = builder.arena_env
    if arena.task is None and arena.embodiment is None:
        return

    from isaaclab.managers import EventTermCfg
    from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

    task_events = arena.task.get_events_cfg() if arena.task is not None else None
    if task_events is not None:
        assert not any(
            isinstance(getattr(task_events, field.name), EventTermCfg) for field in fields(task_events)
        ), "Automatic settled placement does not support custom task events"
    events = arena.embodiment.get_events_cfg() if arena.embodiment is not None else None
    if events is None:
        return
    for field in fields(events):
        term = getattr(events, field.name)
        if not isinstance(term, EventTermCfg):
            continue
        assert (
            term.mode == "reset" and term.func is franka_stack_events.randomize_joint_by_gaussian_offset
        ), f"Automatic settled placement cannot reproduce robot event '{field.name}'"
        term.params = {**term.params, "mean": 0.0, "std": 0.0}
