# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time the normal OpenPI rollout with stock and independent wrist rendering."""

import json
import time
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--renderer", choices=["tiled", "separate"], required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--steps", type=int, default=180)
args = parser.parse_args()

with SimulationAppContext(args):
    import torch

    from isaaclab_arena.evaluation import run_execution
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.experiment_runner import _write_arena_experiment_result
    from isaaclab_arena.evaluation.policy_runner import rollout_policy
    from isaaclab_arena.visualization.report import build_report
    from isaaclab_arena_examples.camera_intrinsics.separate_camera_renderer import SeparateCameraRenderer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    experiment = load_arena_experiment_from_config_file(
        str(Path(__file__).with_name("benchmark.yaml")),
        device=args.device,
        overrides=[
            f"shared.environment_builder.num_envs={args.num_envs}",
            f"shared.rollout_limit.num_steps={args.warmup + args.steps}",
        ],
    )
    original_builder = run_execution.build_arena_builder_from_run_cfg

    def build_builder(cfg):
        builder = original_builder(cfg)
        original_build = builder.build_env_cfg

        def build_env_cfg(*a, **kw):
            env_cfg, env_kwargs = original_build(*a, **kw)
            if args.renderer == "separate":
                env_cfg.scene.wrist_camera.renderer_cfg.class_type = SeparateCameraRenderer
            return env_cfg, env_kwargs

        builder.build_env_cfg = build_env_cfg
        return builder

    run_execution.build_arena_builder_from_run_cfg = build_builder
    records = []
    metadata = {}

    def timed_rollout(env, policy, num_steps, num_episodes):
        camera = env.unwrapped.scene["wrist_camera"]
        metadata.update(
            renderer=type(camera._renderer).__name__,
            camera_type=type(camera).__name__,
            step_dt=env.unwrapped.step_dt,
            resolution=[camera.cfg.width, camera.cfg.height],
            sensor_count=len(env.unwrapped.scene.sensors),
            camera_names=[name for name, sensor in env.unwrapped.scene.sensors.items() if hasattr(sensor.cfg, "width")],
            wrist_render_products=len(camera._render_data.children) if args.renderer == "separate" else 1,
            open_loop_horizon=policy._open_loop_horizon,
        )
        original_get_action = policy.get_action
        original_step = env.step
        pending = {}
        inference_calls = 0
        original_call = policy._call_server_with_retry

        def call(*a, **kw):
            nonlocal inference_calls
            inference_calls += 1
            return original_call(*a, **kw)

        policy._call_server_with_retry = call

        def get_action(*a, **kw):
            torch.cuda.synchronize()
            pending["start"] = time.perf_counter()
            pending["inference_start"] = inference_calls
            action = original_get_action(*a, **kw)
            torch.cuda.synchronize()
            pending["policy_end"] = time.perf_counter()
            return action

        def step(*a, **kw):
            result = original_step(*a, **kw)
            torch.cuda.synchronize()
            end = time.perf_counter()
            records.append(
                dict(
                    step=len(records),
                    policy_s=pending["policy_end"] - pending["start"],
                    env_s=end - pending["policy_end"],
                    end=end,
                    start=pending["start"],
                    inference_calls=inference_calls - pending["inference_start"],
                    resets=int(env.unwrapped.reset_env_ids.numel()),
                )
            )
            if len(records) % 30 == 0:
                print("BENCH_PROGRESS", args.renderer, args.num_envs, len(records), flush=True)
            return result

        policy.get_action = get_action
        env.step = step
        return rollout_policy(env, policy, num_steps, num_episodes)

    run_execution.rollout_policy = timed_rollout
    results = run_execution.execute_experiment(experiment, out)
    _write_arena_experiment_result(experiment, results, out)
    build_report(out)
    measured = records[args.warmup :]
    assert len(measured) == args.steps
    elapsed = measured[-1]["end"] - measured[0]["start"]
    summary = dict(
        status="completed",
        mode=args.renderer,
        num_envs=args.num_envs,
        warmup_steps=args.warmup,
        measured_steps=args.steps,
        elapsed_s=elapsed,
        vector_steps_per_s=args.steps / elapsed,
        env_steps_per_s=args.num_envs * args.steps / elapsed,
        mean_policy_ms=1000 * sum(r["policy_s"] for r in measured) / len(measured),
        mean_env_ms=1000 * sum(r["env_s"] for r in measured) / len(measured),
        inference_calls=sum(r["inference_calls"] for r in measured),
        resets=sum(r["resets"] for r in measured),
        **metadata,
    )
    (out / "benchmark.json").write_text(json.dumps(dict(summary=summary, steps=records), indent=2))
    print("BENCH_RESULT", json.dumps(summary), flush=True)
