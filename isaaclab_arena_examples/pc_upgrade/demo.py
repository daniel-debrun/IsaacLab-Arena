# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the PC upgrade scene and its explicitly assisted integration demonstration.

The demonstration uses bounded virtual-grasp forces and a driven panel hinge.
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
    parser.add_argument("--video", action="store_true", help="Record a 30 fps multi-angle MP4 (one environment).")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Pace the demonstration at wall-clock speed.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.capture or args.video
    assert not args.video or args.num_envs == 1, "Video recording requires one environment"
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        run(args)


def run(args):
    import json
    import time
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena_environments.pc_upgrade.layout import BIN_POSITION, PC_POSITION, SLOT_IN_PC, SLOT_QUATERNION
    from isaaclab_arena_environments.pc_upgrade_environment import PCUpgradeEnvironment, PCUpgradeEnvironmentCfg
    from isaaclab_arena_examples.pc_upgrade.motion import AssistedMotionCfg

    args.output.mkdir(parents=True, exist_ok=True)
    description = PCUpgradeEnvironment().build(PCUpgradeEnvironmentCfg(enable_cameras=args.capture or args.video))
    video = None
    video_success = False
    if args.video:
        description.embodiment.camera_config.overview_cam.width = 1440
        description.embodiment.camera_config.overview_cam.height = 1080

    def show(view, title, hold=0.0):
        if video is not None:
            video.shot(view, title)
            video.hold(hold)

    def record_terminal(finishing_env, env_id):
        if env_id == 0 and finishing_env.progress_tracker.is_complete()[env_id]:
            if video is not None:
                video.shot("Overview", "Complete: RTX installed, chassis closed, mock cards recycled")
                video.hold(3.0)
                video.active = False
            snapshot("07_success")
        return {}

    description.episode_recorder_terms["pc_demo_terminal"] = EpisodeRecorderTermCfg(func=record_terminal)
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(
            num_envs=args.num_envs,
            env_spacing=3.0,
            solve_relations=False,
            device=args.device,
        ),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.actions.assisted_motion = AssistedMotionCfg()
    if args.video:
        from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg

        from isaaclab_arena_examples.pc_upgrade.video import VideoRecorderTerm

        env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
        env_cfg.recorders.video = RecorderTermCfg(class_type=VideoRecorderTerm)
    # Only this demonstration adds a position drive; policy environments retain a passive door.
    hinge = env_cfg.scene.pc_chassis.actuators["hinge"]
    hinge.stiffness = 30.0
    hinge.damping = 8.0
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()
    base = env.unwrapped
    if (args.capture or args.video) and not base.sim.has_gui:
        # Headless snapshots render explicitly; avoid rendering unused camera frames.
        base.render_enabled = False
    if args.video:
        from isaaclab_arena_examples.pc_upgrade.video import TaskVideo

        video = TaskVideo(base, args.output)
        base.pc_upgrade_video = video
    base.episode_recorder.set_output_path(args.output / "episodes.jsonl")
    action = torch.zeros(env.action_space.shape, device=base.device)
    action[:, -1] = 1.0
    motion = base.action_manager.get_term("assisted_motion")
    evidence = []
    succeeded = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)

    def step(count=1):
        for _ in range(count):
            started = time.monotonic()
            _, _, terminated, truncated, _ = env.step(action)
            success = base.termination_manager.get_term("success")
            succeeded.logical_or_(success)
            assert not (terminated & ~success).any(), "Unexpected task failure during assisted demonstration"
            assert not truncated.any(), "Demonstration exceeded the task time limit"
            if args.real_time:
                time.sleep(max(0.0, base.step_dt - (time.monotonic() - started)))
            if succeeded.all():
                return

    def snapshot(label):
        state = {
            "stage": label,
            "completed": base.progress_tracker.get_subtask_completion().tolist(),
        }
        for name in ("rtx_5090", "vortex_vx880", "cinder_c700"):
            state[name] = (base.scene[name].data.root_pos_w - base.scene.env_origins).tolist()
            state[f"{name}_quaternion"] = base.scene[name].data.root_quat_w[:].tolist()
        state["panel_angle"] = base.scene["pc_chassis"].data.joint_pos.tolist()
        evidence.append(state)
        print(json.dumps(state), flush=True)
        if args.capture and video is None:
            from PIL import Image

            for _ in range(32):
                base.sim.render()
            base.scene["overview_cam"].update(base.step_dt, force_recompute=True)
            pixels = base.scene["overview_cam"].data.output["rgb"][0, :, :, :3].cpu().numpy()
            assert pixels is not None and pixels.size, "Renderer did not return pixels"
            Image.fromarray(pixels).save(args.output / f"{label}.png")

    def interior_snapshot(label):
        if not args.capture or video is not None:
            return
        from isaaclab_arena_environments.pc_upgrade.cameras import EYE, TARGET

        camera = base.scene["overview_cam"]
        camera.set_world_poses_from_view(
            torch.tensor([[0.96, -0.20, 1.32]], device=base.device) + base.scene.env_origins,
            torch.tensor([[0.74, 0.40, 1.02]], device=base.device) + base.scene.env_origins,
        )
        snapshot(label)
        camera.set_world_poses_from_view(
            torch.tensor([EYE], device=base.device) + base.scene.env_origins,
            torch.tensor([TARGET], device=base.device) + base.scene.env_origins,
        )

    def panel(angle, steps=70):
        motion.move_panel(angle, steps * base.step_dt)
        step(steps + 20)

    def move_card(name, target, steps=60, quaternion=None, free_space=True):
        gpu = base.scene[name]
        target = torch.tensor(target, device=base.device).repeat(base.num_envs, 1) + base.scene.env_origins
        rotation = gpu.data.root_quat_w[:].clone()
        if quaternion is not None:
            rotation[:] = rotation.new_tensor(quaternion)
        motion.move(name, target, rotation, steps * base.step_dt, free_space=free_space)
        step(steps + 15)
        print("PC_UPGRADE_MOTION " + json.dumps(motion.telemetry[-1]), flush=True)

    try:
        step(45)
        snapshot("00_initial")
        show("Overview", "PC upgrade workbench", 2.5)
        show("Top-down", "Identify NVIDIA RTX 5090, VORTEX VX 880 and CINDER C700", 5.0)
        if args.capture and video is None:
            from isaaclab_arena_environments.pc_upgrade.cameras import EYE, TARGET

            camera = base.scene["overview_cam"]
            camera.set_world_poses_from_view(
                torch.tensor([[0.67, -0.67, 1.16]], device=base.device) + base.scene.env_origins,
                torch.tensor([[0.48, -0.32, 0.78]], device=base.device) + base.scene.env_origins,
            )
            snapshot("00_rtx_detail")
            camera.set_world_poses_from_view(
                torch.tensor([[0.42, -0.85, 1.45]], device=base.device) + base.scene.env_origins,
                torch.tensor([[0.35, -0.45, 0.78]], device=base.device) + base.scene.env_origins,
            )
            snapshot("00_cards_detail")
            camera.set_world_poses_from_view(
                torch.tensor([EYE], device=base.device) + base.scene.env_origins,
                torch.tensor([TARGET], device=base.device) + base.scene.env_origins,
            )
        assert not base.progress_tracker.get_subtask_completion().any(), "Initial scene must not satisfy any stage"
        if not args.smoke:
            show("Overview", "1. Open the hinged chassis panel")
            panel(-1.85)
            snapshot("01_open")
            interior_snapshot("01_socket_detail")
            show("Computer", "Motherboard, PCIe socket and GPU support brackets", 3.0)
            show("Top-down", "2. Select and lift the NVIDIA RTX 5090")
            start = base.scene["rtx_5090"].data.root_pos_w[0] - base.scene.env_origins[0]
            move_card("rtx_5090", (start[0].item(), start[1].item(), 1.07))
            snapshot("02_selected")
            show("Overview", "Rotate the card and approach the open side")
            slot = tuple(a + b for a, b in zip(PC_POSITION, SLOT_IN_PC))
            # Rotate in front of the tower, then enter through its open side at slot height.
            move_card("rtx_5090", (slot[0], start[1].item(), 1.07), quaternion=SLOT_QUATERNION)
            move_card("rtx_5090", (slot[0], slot[1] - 0.24, slot[2]))
            snapshot("02_aligned")
            show("Computer", "3. Insert the card into the motherboard PCIe socket", 1.5)
            move_card("rtx_5090", slot, free_space=False)
            motion.release()
            step(45)
            snapshot("03_installed")
            interior_snapshot("03_installed_detail")
            show("Computer", "RTX 5090 seated and released onto its supports", 3.0)
            show("Overview", "4. Close the chassis panel")
            panel(0.0)
            snapshot("04_closed")
            for index, name in enumerate(("vortex_vx880", "cinder_c700")):
                title = "VORTEX VX 880" if index == 0 else "CINDER C700"
                show("Overview", f"{5 + index}. Recycle the fictional {title}")
                current = base.scene[name].data.root_pos_w[0] - base.scene.env_origins[0]
                # Carry above the robot and tower, then lower vertically over the bin.
                move_card(name, (current[0].item(), current[1].item(), 1.60))
                move_card(name, (BIN_POSITION[0], BIN_POSITION[1], 1.60))
                move_card(name, (BIN_POSITION[0], BIN_POSITION[1], 1.25))
                # Release above the rim; the card falls and settles under gravity.
                motion.release()
                step(100)
                if not succeeded.all():
                    snapshot(f"0{5 + index}_recycled_{name}")
            assert succeeded.all(), f"Task did not succeed: {base.progress_tracker.get_subtask_completion()}"
        if args.smoke:
            env.reset()
        assert not base.progress_tracker.get_subtask_completion().any(), "Reset left stale task progress"
        result = {
            "mode": "smoke" if args.smoke else "assisted_force_demonstration",
            "num_envs": base.num_envs,
            "success": succeeded.tolist(),
            "reset_verified": True,
            "evidence": evidence,
            "motion_telemetry": motion.telemetry,
        }
        _validate_motion(motion.telemetry)
        video_success = args.smoke or bool(succeeded.all())
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            "PC_UPGRADE_RESULT " + json.dumps({key: value for key, value in result.items() if key != "evidence"}),
            flush=True,
        )
    finally:
        try:
            if video is not None:
                video.finish(video_success)
        finally:
            env.close()


def _validate_motion(telemetry):
    """Check that virtual grasp tracking stayed smooth throughout the episode."""
    for metrics in telemetry:
        if metrics["free_space"]:
            assert metrics["max_tracking_error_m"] < 0.002, f"Unstable virtual grasp: {metrics}"
            assert metrics["max_orientation_error_rad"] < 0.01, f"Unstable card orientation: {metrics}"
            assert metrics["max_step_displacement_m"] < 0.008, f"Discontinuous card motion: {metrics}"


if __name__ == "__main__":
    main()
