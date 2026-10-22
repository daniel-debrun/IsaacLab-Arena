# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture real RGB frames while exercising Arena's intrinsics reset event."""

import json
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--capture-dir", required=True)
args = parser.parse_args()
assert args.num_envs in (1, 4), "This capture sequence supports one or four environments"

with SimulationAppContext(args):
    import numpy as np
    import torch

    from PIL import Image

    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.variations.uniform_sampler import UniformSampler

    out = Path(args.capture_dir)
    out.mkdir(parents=True, exist_ok=False)
    experiment = load_arena_experiment_from_config_file(
        str(Path(__file__).with_name("capture.yaml")),
        device=args.device,
        overrides=[f"shared.environment_builder.num_envs={args.num_envs}"],
    )
    cfg = experiment.runs["intrinsics"]
    prefix = "droid_rel_joint_pos.camera_intrinsics_wrist_camera"
    cfg.variations[prefix + ".sampler_cfg.low"] = [0.0, 0.0]
    cfg.variations[prefix + ".sampler_cfg.high"] = [0.0, 0.0]
    builder = build_arena_builder_from_run_cfg(cfg)
    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    env_cfg, env_kwargs = builder.build_env_cfg()
    for name in builder.arena_env.embodiment.camera_config.camera_names():
        assert getattr(env_cfg.scene, name).renderer_cfg.class_type is SeparateCameraRenderer
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()
    base = env.unwrapped
    camera = base.scene["wrist_camera"]
    term = base.event_manager.get_term_cfg("wrist_camera_intrinsics_variation")
    sampler = term.params["sampler"]
    nominal = camera.data.intrinsic_matrices.torch.clone()
    env_ids = torch.arange(args.num_envs, device=base.device)
    records = []

    def capture(label):
        # Render without stepping physics, so geometry and camera extrinsics stay fixed.
        for _ in range(30):
            base.sim.render()
            camera.update(base.step_dt, force_recompute=True)
            _ = camera.data.output["rgb"]
        rgb = camera.data.output["rgb"].torch.cpu().numpy()[..., :3]
        K = camera.data.intrinsic_matrices.torch.cpu().numpy()
        for i in range(args.num_envs):
            Image.fromarray(rgb[i].astype(np.uint8)).save(out / f"{label}_env{i}.png")
        records.append({
            "label": label,
            "intrinsic_matrices": K.tolist(),
            "focal_scales_y": (K[:, 1, 1] / nominal[:, 1, 1].cpu().numpy()).tolist(),
            "focal_scales": (K[:, 0, 0] / nominal[:, 0, 0].cpu().numpy()).tolist(),
            "camera_t_W": camera.data.pos_w.torch.cpu().tolist(),
            "camera_q_W_C": camera.data.quat_w_world.torch.cpu().tolist(),
        })
        print("CAPTURE", label, records[-1]["focal_scales"], flush=True)

    def apply_and_capture(label, deltas, selected=None):
        ids = env_ids if selected is None else torch.tensor(selected, device=base.device)
        # Use Arena's sampler replay interface for reproducible input values.
        # A separate sampler avoids duplicate episode-record entries during this
        # diagnostic; the installed event still performs all calibration writes.
        controlled_sampler = UniformSampler(low=[0.0, 0.0], high=[0.0, 0.0])
        controlled_sampler.set_replay_sampler(
            lambda count, selected_ids: [[deltas[int(i)], deltas[int(i)]] for i in selected_ids]
        )
        term.func(base, ids, asset_cfg=term.params["asset_cfg"], sampler=controlled_sampler)
        capture(label)

    capture("baseline")
    apply_and_capture("wide", [-0.4] * args.num_envs)
    apply_and_capture("tele", [0.5] * args.num_envs)
    apply_and_capture("restored", [0.0] * args.num_envs)
    if args.num_envs > 1:
        apply_and_capture("mixed", [-0.4, 0.0, 0.25, 0.5])
        apply_and_capture("env0_only", [0.5, 0.0, 0.25, 0.5], selected=[0])
    # Exercise ordinary env.reset as well, with independent seeded uniform samples.
    sampler.set_replay_sampler(None)
    sampler.low = torch.tensor([-0.45, -0.45])
    sampler.high = torch.tensor([0.6, 0.6])
    sampler.set_generator(torch.Generator().manual_seed(42))
    for reset_index in range(2):
        env.reset()
        capture(f"random_reset{reset_index}")
    (out / "measurements.json").write_text(
        json.dumps(
            {
                "render_products": [
                    {"path": str(child.render_product.path), "cameras": list(child.spec.camera_prim_paths)}
                    for child in camera._render_data.children
                ],
                "renderer_type": type(camera._renderer).__name__,
                "num_envs": args.num_envs,
                "physics": "physx",
                "camera_type": type(camera).__name__,
                "nominal_intrinsic_matrices": nominal.cpu().tolist(),
                "captures": records,
                "method": (
                    "Fixed-scene captures call the installed Arena reset event with replayed sampler values;"
                    " random_reset captures use ordinary env.reset and the real uniform sampler."
                ),
            },
            indent=2,
        )
    )
    env.close()
