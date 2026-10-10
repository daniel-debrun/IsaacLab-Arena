# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the reference coffee-disposal sequence with explicit force assistance."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/coffee_disposal"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.capture or args.video
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        Demo(args).run()


def record_demo_terminal(env, env_id):
    """Resolve the live demo after Arena copies the recorder configuration."""
    demo = getattr(env, "coffee_demo", None)
    if demo is not None:
        return demo.terminal(env, env_id)
    return {}


class Demo:
    """Execute physical manipulations and preserve stage evidence before automatic reset."""

    def __init__(self, args):
        import torch

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
        from isaaclab_arena_environments.coffee_disposal_environment import (
            CoffeeDisposalEnvironment,
            CoffeeDisposalEnvironmentCfg,
        )
        from isaaclab_arena_examples.coffee_disposal.motion import CoffeeMotionCfg, configure_demo_drives

        self.args = args
        args.output.mkdir(parents=True, exist_ok=True)
        self.video = None
        self.evidence = []
        description = CoffeeDisposalEnvironment().build(
            CoffeeDisposalEnvironmentCfg(enable_cameras=args.enable_cameras)
        )
        description.episode_recorder_terms["coffee_demo_terminal"] = EpisodeRecorderTermCfg(func=record_demo_terminal)
        builder = ArenaEnvBuilder(
            description,
            ArenaEnvBuilderCfg(num_envs=args.num_envs, env_spacing=12.0, solve_relations=False, device=args.device),
        )
        cfg, kwargs = builder.compose_manager_cfg()
        cfg.actions.coffee_motion = CoffeeMotionCfg()
        configure_demo_drives(cfg)
        if args.video:
            from isaaclab.managers import DatasetExportMode, RecorderTermCfg

            from isaaclab_arena.progress_tracking.progress_tracker import ProgressTrackingRecorderManagerCfg
            from isaaclab_arena_examples.coffee_disposal.video import VideoRecorderTerm

            cfg.recorders = ProgressTrackingRecorderManagerCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
            cfg.recorders.video = RecorderTermCfg(class_type=VideoRecorderTerm)
        self.env = builder.make_registered(cfg, kwargs)
        self.env.reset()
        self.base = self.env.unwrapped
        self.base.coffee_demo = self
        if args.enable_cameras and not self.base.sim.has_gui:
            self.base.render_enabled = False
        self.base.episode_recorder.set_output_path(args.output / "episodes.jsonl")
        self.action = torch.zeros(self.env.action_space.shape, device=self.base.device)
        self.action[:, -1] = 1.0
        self.motion = self.base.action_manager.get_term("coffee_motion")
        self.succeeded = torch.zeros(self.base.num_envs, dtype=torch.bool, device=self.base.device)
        if args.video:
            from isaaclab_arena_examples.coffee_disposal.video import TaskVideo

            self.video = TaskVideo(self.base, args.output)
            self.base.coffee_video = self.video

    def step(self, seconds):
        import time

        for _ in range(round(seconds / self.base.step_dt)):
            started = time.monotonic()
            _, _, terminated, truncated, _ = self.env.step(self.action)
            success = self.base.termination_manager.get_term("success")
            self.succeeded.logical_or_(success)
            assert not (terminated & ~success).any(), "Physical task failure; inspect the stage evidence"
            assert not truncated.any(), "Coffee disposal exceeded the episode time limit"
            if self.args.real_time:
                time.sleep(max(0.0, self.base.step_dt - (time.monotonic() - started)))
            if self.succeeded.all():
                return

    def show(self, view, title, hold=0.0):
        if self.video is not None:
            self.video.shot(view, title)
            self.video.hold(hold)
        elif self.args.capture:
            import torch

            from isaaclab_arena_environments.coffee_disposal.cameras import VIEWS

            eye, target = VIEWS[view]
            self.base.scene["overview_cam"].set_world_poses_from_view(
                torch.tensor([eye], device=self.base.device) + self.base.scene.env_origins,
                torch.tensor([target], device=self.base.device) + self.base.scene.env_origins,
            )

    def snapshot(self, label, expected_stages=0):
        import json

        from isaaclab_arena_environments.coffee_disposal.layout import GROUND_NAMES

        record = {"stage": label, "completed": self.base.progress_tracker.get_subtask_completion().tolist()}
        for name in ("grounds_container", "bin_lid", *GROUND_NAMES):
            data = self.base.scene[name].data
            record[name] = (data.root_pos_w - self.base.scene.env_origins).tolist()
            record[name + "_quaternion"] = data.root_quat_w[:].tolist()
            record[name + "_velocity"] = data.root_vel_w[:].tolist()
        for name in ("coffee_machine", "waste_cabinet"):
            record[name + "_joints"] = self.base.scene[name].data.joint_pos[:].tolist()
        self.evidence.append(record)
        print("COFFEE_STAGE " + json.dumps(record), flush=True)
        if self.args.capture and self.video is None:
            from PIL import Image

            for _ in range(24):
                self.base.sim.render()
            camera = self.base.scene["overview_cam"]
            camera.update(self.base.step_dt, force_recompute=True)
            Image.fromarray(camera.data.output["rgb"][0, :, :, :3].cpu().numpy()).save(
                self.args.output / f"{label}.png"
            )

        assert all(
            all(row[:expected_stages]) for row in record["completed"]
        ), f"Stage {label} did not complete: {record['completed']}"

    def terminal(self, env, env_id):
        from isaaclab_arena_environments.coffee_disposal.layout import GROUND_NAMES

        if env_id == 0 and env.progress_tracker.is_complete()[env_id]:
            self.show("Overview", "Complete: grounds disposed; machine, lid and drawer restored", 4.0)
            self.snapshot("11_success")
            if self.video is not None:
                self.video.active = False
        # Read the finishing environment directly before its reset, independently
        # of extras or the timing of other environments' completion.
        objects = {}
        for name in ("grounds_container", "bin_lid", *GROUND_NAMES):
            state = env.scene[name].data.root_state_w[env_id].clone()
            state[:3] -= env.scene.env_origins[env_id]
            objects[name] = state.tolist()
        joints = {}
        for name in ("coffee_machine", "waste_cabinet"):
            joints[name] = env.scene[name].data.joint_pos[env_id].tolist()
        return {
            "coffee_terminal_state": {
                "completed": env.progress_tracker.get_subtask_completion()[env_id].tolist(),
                "objects": objects,
                "joints": joints,
            }
        }

    def joint(self, asset, name, target, duration=3.0):
        self.motion.move_joint(asset, name, target, duration)
        self.step(duration + 1.0)

    def move(self, name, position, quaternion=(0.0, 0.0, 0.0, 1.0), duration=2.5, free_space=True):
        import torch

        target = (
            torch.tensor(position, device=self.base.device).repeat(self.base.num_envs, 1) + self.base.scene.env_origins
        )
        rotation = torch.tensor(quaternion, device=self.base.device).repeat(self.base.num_envs, 1)
        self.motion.move(name, target, rotation, duration, free_space=free_space)
        self.step(duration + 0.6)
        import json

        print("COFFEE_MOTION " + json.dumps(self.motion.telemetry[-1]), flush=True)

    def release(self, settle=1.2):
        self.motion.release()
        self.step(settle)

    def sequence(self):
        import math

        from isaaclab_arena_environments.coffee_disposal.layout import (
            BIN_IN_DRAWER,
            CABINET_POSITION,
            CUP_HOME,
            DRAWER_EXTENSION,
            LID_IN_DRAWER,
            LID_PARK_IN_DRAWER,
            SPOUT_RAISED,
            TRAY_EXTENSION,
        )

        cup = (CUP_HOME[0], CUP_HOME[1] + TRAY_EXTENSION, CUP_HOME[2])
        lid = tuple(
            a + b + (DRAWER_EXTENSION if i == 1 else 0) for i, (a, b) in enumerate(zip(CABINET_POSITION, LID_IN_DRAWER))
        )
        park = tuple(
            a + b + (DRAWER_EXTENSION if i == 1 else 0)
            for i, (a, b) in enumerate(zip(CABINET_POSITION, LID_PARK_IN_DRAWER))
        )
        bin_x = CABINET_POSITION[0] + BIN_IN_DRAWER[0]
        bin_y = CABINET_POSITION[1] + BIN_IN_DRAWER[1] + DRAWER_EXTENSION
        above = (bin_x, bin_y, 0.86)
        upright = (0.0, 0.0, 0.0, 1.0)
        self.show("Machine", "1. Raise the height-adjustable dispenser")
        self.joint("coffee_machine", "spout_slide", SPOUT_RAISED, 1.5)
        self.snapshot("01_dispenser_raised", 1)
        self.show("Machine", "2. Pull out the drip tray with the grounds container")
        self.joint("coffee_machine", "tray_slide", TRAY_EXTENSION, 4.0)
        self.show("Top-down", "The removable grounds container rides on the drip tray", 2.0)
        self.snapshot("02_tray_extracted", 2)
        self.show("Overview", "3. Open the under-counter waste drawer")
        self.joint("waste_cabinet", "drawer_slide", DRAWER_EXTENSION, 3.5)
        self.show("Waste drawer", "4. Remove the organic-bin lid and park it on the adjacent rim")
        self.snapshot("03_drawer_open", 3)
        self.move("bin_lid", (lid[0], lid[1], lid[2] + 0.12), duration=1.5)
        self.move("bin_lid", (park[0], park[1], park[2] + 0.12), duration=2.0)
        self.move("bin_lid", (park[0], park[1], park[2] + 0.004), duration=1.5, free_space=False)
        self.release()
        self.snapshot("04_lid_parked", 4)
        self.show("Machine", "5. Lift the separate grounds container from the tray")
        self.move("grounds_container", (cup[0], cup[1], 1.28), duration=2.5)
        self.snapshot("05_container_lifted", 5)
        self.show("Overview", "6. Carry the grounds upright to the open organic bin")
        self.move("grounds_container", (cup[0], -0.80, 1.30), duration=3.0)
        self.move("grounds_container", (bin_x, bin_y, 1.15), duration=3.0)
        self.move("grounds_container", above, duration=2.0)
        self.show("Grounds close-up", "7. Tip and rock the container; let the grounds fall into the bin", 2.0)
        self.snapshot("06_above_bin", 6)
        for angle, duration in ((155, 3.0), (140, 0.8), (170, 0.8), (145, 0.8), (165, 0.8)):
            radians = math.radians(angle)
            self.move(
                "grounds_container", above, (math.sin(radians / 2), 0, 0, math.cos(radians / 2)), duration=duration
            )
        # Different clump piles take different times to stop rocking. Keep the
        # physical settled criteria unchanged and wait for their observed result.
        self.step(1.5)
        for _ in range(20):
            if self.base.progress_tracker.get_subtask_completion()[:, 6].all():
                break
            self.step(0.5)
        self.snapshot("07_grounds_disposed", 7)
        self.move("grounds_container", above, upright, duration=2.5)
        self.show("Overview", "8. Return the upright container to its seat in the drip tray")
        self.move("grounds_container", (bin_x, bin_y, 1.30), duration=2.0)
        self.move("grounds_container", (cup[0], -0.80, 1.30), duration=3.0)
        self.move("grounds_container", (cup[0], cup[1], 1.20), duration=3.0)
        self.show("Top-down", "Align, lower and release the container into its tray recess")
        self.move("grounds_container", (cup[0], cup[1], cup[2] + 0.004), duration=2.5, free_space=False)
        self.release(1.5)
        self.snapshot("08_container_returned", 8)
        self.show("Bin interior", "Loose grounds retained inside the organic bin", 2.0)
        self.snapshot("08_bin_contents")
        self.show("Machine", "9. Slide the complete tray assembly back into the machine")
        self.joint("coffee_machine", "tray_slide", 0.0, 4.0)
        self.snapshot("09_tray_home", 9)
        self.show("Waste drawer", "10. Retrieve and reseat the organic-bin lid")
        self.move("bin_lid", (park[0], park[1], park[2] + 0.12), duration=1.5)
        self.move("bin_lid", (lid[0], lid[1], lid[2] + 0.12), duration=2.0)
        self.move("bin_lid", (lid[0], lid[1], lid[2] + 0.004), duration=1.5, free_space=False)
        self.release(1.5)
        self.snapshot("10_lid_replaced", 10)
        self.show("Overview", "11. Close the waste drawer; verify the complete restored scene")
        self.joint("waste_cabinet", "drawer_slide", 0.0, 4.0)
        self.step(2.0)
        assert self.succeeded.all(), f"Incomplete task: {self.base.progress_tracker.get_subtask_completion()}"

    def run(self):
        import json

        valid = False
        try:
            self.step(3.0)
            self.show("Overview", "Dispose of spent coffee grounds and restore the kitchen", 3.0)
            self.snapshot("00_overview")
            if self.video is not None:
                self.video.show_plan = True
                self.show("Overview", "Before we start: the complete task sequence", 10.0)
                self.video.show_plan = False
            if not self.args.smoke:
                self.sequence()
            else:
                self.show("Machine", "Automatic coffee machine and removable grounds tray", 2.0)
                self.snapshot("00_machine")
                self.env.reset()
            assert not self.base.progress_tracker.get_subtask_completion().any(), "Reset left stale progress"
            report = {
                "mode": "smoke" if self.args.smoke else "assisted_force_demonstration",
                "num_envs": self.base.num_envs,
                "success": self.succeeded.tolist(),
                "reset_verified": True,
                "evidence": self.evidence,
                "motion_telemetry": self.motion.telemetry,
            }
            (self.args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print("COFFEE_RESULT " + json.dumps({k: v for k, v in report.items() if k != "evidence"}), flush=True)
            valid = True
        finally:
            try:
                if self.video is not None:
                    self.video.finish(valid, bool(self.succeeded.all()))
            finally:
                self.env.close()
        if self.video is not None and valid:
            from isaaclab_arena_examples.coffee_disposal.video import make_speed_versions

            make_speed_versions(self.args.output)


if __name__ == "__main__":
    main()
