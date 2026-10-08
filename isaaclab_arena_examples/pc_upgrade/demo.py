# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the PC upgrade scene and its explicitly assisted integration demonstration.

The demonstration writes object poses and the panel joint state along trajectories.
It validates scene physics, ordered task predicates and resets, not autonomous
robot manipulation. Ordinary policies use the registered pc_upgrade environment.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Step initial scene only.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/arena-pc-upgrade-output"))
    parser.add_argument("--capture", action="store_true", help="Save rendered stage snapshots.")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.capture
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        run(args)


def run(args):
    import json
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena_environments.pc_upgrade.layout import BIN_POSITION, PC_POSITION, SLOT_IN_PC
    from isaaclab_arena_environments.pc_upgrade_environment import PCUpgradeEnvironment, PCUpgradeEnvironmentCfg

    args.output.mkdir(parents=True, exist_ok=True)
    description = PCUpgradeEnvironment().build(PCUpgradeEnvironmentCfg(enable_cameras=args.capture))

    def record_terminal(finishing_env, env_id):
        if env_id == 0 and finishing_env.progress_tracker.is_complete()[env_id]:
            snapshot("07_success")
        return {}

    description.episode_recorder_terms["pc_demo_terminal"] = EpisodeRecorderTermCfg(func=record_terminal)
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(num_envs=args.num_envs, env_spacing=3.0, solve_relations=False, device=args.device),
    )
    env = builder.make_registered()
    env.reset()
    base = env.unwrapped
    base.episode_recorder.set_output_path(args.output / "episodes.jsonl")
    action = torch.zeros(env.action_space.shape, device=base.device)
    action[:, -1] = 1.0
    evidence = []
    succeeded = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)

    def step(count=1):
        for _ in range(count):
            _, _, terminated, truncated, _ = env.step(action)
            success = base.termination_manager.get_term("success")
            succeeded.logical_or_(success)
            assert not (terminated & ~success).any(), "Unexpected task failure during assisted demonstration"
            assert not truncated.any(), "Demonstration exceeded the task time limit"
            if succeeded.all():
                return

    def snapshot(label):
        state = {"stage": label, "completed": base.progress_tracker.get_subtask_completion().tolist()}
        for name in ("rtx_5090", "vortex_vx880", "cinder_c700"):
            state[name] = (base.scene[name].data.root_pos_w - base.scene.env_origins).tolist()
        state["panel_angle"] = base.scene["pc_chassis"].data.joint_pos.tolist()
        evidence.append(state)
        print(json.dumps(state), flush=True)
        if args.capture:
            from PIL import Image

            for _ in range(8):
                base.sim.render()
            base.scene["overview_cam"].update(base.step_dt, force_recompute=True)
            pixels = base.scene["overview_cam"].data.output["rgb"][0, :, :, :3].cpu().numpy()
            assert pixels is not None and pixels.size, "Renderer did not return pixels"
            Image.fromarray(pixels).save(args.output / f"{label}.png")

    def panel(angle, steps=100):
        pc = base.scene["pc_chassis"]
        initial = pc.data.joint_pos.clone()
        for index in range(steps):
            fraction = (index + 1) / steps
            target = initial * (1 - fraction) + angle * fraction
            pc.write_joint_state_to_sim(target, torch.zeros_like(target))
            step()
        step(20)

    def move_card(name, target, steps=60):
        gpu = base.scene[name]
        initial = gpu.data.root_pos_w.clone()
        target = initial.new_tensor(target) + base.scene.env_origins
        for index in range(steps):
            fraction = (index + 1) / steps
            position = initial.lerp(target, fraction)
            rotation = position.new_tensor([0.0, 0.0, 0.0, 1.0]).repeat(base.num_envs, 1)
            gpu.write_root_pose_to_sim(torch.cat((position, rotation), dim=-1))
            gpu.write_root_velocity_to_sim(torch.zeros((base.num_envs, 6), device=base.device))
            step()

    try:
        step(45)
        snapshot("00_initial")
        if args.capture:
            from isaaclab_arena_environments.pc_upgrade.cameras import EYE, TARGET

            camera = base.scene["overview_cam"]
            camera.set_world_poses_from_view(
                torch.tensor([[0.58, -0.60, 1.10]], device=base.device) + base.scene.env_origins,
                torch.tensor([[0.42, -0.25, 0.78]], device=base.device) + base.scene.env_origins,
            )
            snapshot("00_rtx_detail")
            camera.set_world_poses_from_view(
                torch.tensor([[0.42, -0.70, 1.38]], device=base.device) + base.scene.env_origins,
                torch.tensor([[0.35, -0.36, 0.78]], device=base.device) + base.scene.env_origins,
            )
            snapshot("00_cards_detail")
            camera.set_world_poses_from_view(
                torch.tensor([EYE], device=base.device) + base.scene.env_origins,
                torch.tensor([TARGET], device=base.device) + base.scene.env_origins,
            )
        assert not base.progress_tracker.get_subtask_completion().any(), "Initial scene must not satisfy any stage"
        if not args.smoke:
            panel(-1.85)
            snapshot("01_open")
            start = base.scene["rtx_5090"].data.root_pos_w[0] - base.scene.env_origins[0]
            move_card("rtx_5090", (start[0].item(), start[1].item(), 1.10))
            snapshot("02_selected")
            slot = tuple(a + b for a, b in zip(PC_POSITION, SLOT_IN_PC))
            move_card("rtx_5090", (slot[0], slot[1], 1.13))
            move_card("rtx_5090", slot)
            step(45)
            snapshot("03_installed")
            panel(0.0)
            snapshot("04_closed")
            for index, name in enumerate(("vortex_vx880", "cinder_c700")):
                current = base.scene[name].data.root_pos_w[0] - base.scene.env_origins[0]
                move_card(name, (current[0].item(), current[1].item(), 1.25))
                move_card(name, (BIN_POSITION[0], BIN_POSITION[1], 1.25))
                # Release above the rim; the card falls and settles under gravity.
                step(100)
                if not succeeded.all():
                    snapshot(f"0{5 + index}_recycled_{name}")
            assert succeeded.all(), f"Task did not succeed: {base.progress_tracker.get_subtask_completion()}"
        if args.smoke:
            env.reset()
        assert not base.progress_tracker.get_subtask_completion().any(), "Reset left stale task progress"
        result = {
            "mode": "smoke" if args.smoke else "assisted_state_demonstration",
            "num_envs": base.num_envs,
            "success": succeeded.tolist(),
            "reset_verified": True,
            "evidence": evidence,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            "PC_UPGRADE_RESULT " + json.dumps({key: value for key, value in result.items() if key != "evidence"}),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
