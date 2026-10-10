# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record an edited, multi-angle MP4 directly from the demonstration's physics steps."""

import json
import shutil
import subprocess
import torch
from pathlib import Path

from isaaclab.managers import RecorderTerm
from PIL import Image, ImageDraw, ImageFont

from isaaclab_arena_environments.coffee_disposal.cameras import VIEWS


class VideoRecorderTerm(RecorderTerm):
    """Record at 30 Hz without changing the physics or policy control rates."""

    def record_post_physics_decimation_step(self):
        video = getattr(self._env, "coffee_video", None)
        if video is not None and video.active and self._env._sim_step_counter % video.stride == 0:
            video.capture()
        return None, None


class TaskVideo:
    """Stream labelled RGB frames to an MP4 encoder, with explicit shots and holds."""

    def __init__(self, env, directory: Path, fps=30):
        self.env = env
        self.directory = directory
        self.path = directory / "coffee_disposal_1x.mp4"
        self.partial = directory / "coffee_disposal_1x.partial.mp4"
        self.camera = env.scene["overview_cam"]
        self.fps = fps
        self.stride = round(1.0 / (env.physics_dt * fps))
        assert abs(self.stride * env.physics_dt - 1.0 / fps) < 1e-8
        self.active = False
        self.frames = 0
        self.shots = []
        self.view = "Overview"
        self.title = "Coffee grounds disposal"
        self.last_frame = None
        self.show_plan = False
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.font = ImageFont.truetype(font, 27)
        self.small_font = ImageFont.truetype(font, 19)
        self.log = (directory / "video_encoder.log").open("w")
        encoder = shutil.which("ffmpeg")
        assert encoder is not None, "Video capture requires ffmpeg in PATH"
        self.encoder = subprocess.Popen(
            [
                encoder,
                "-y",
                "-loglevel",
                "warning",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.camera.cfg.width}x{self.camera.cfg.height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.partial),
            ],
            stdin=subprocess.PIPE,
            stderr=self.log,
            stdout=subprocess.DEVNULL,
        )

    def shot(self, view, title):
        """Cut to a named camera angle and update the on-screen action caption."""
        self.active = False
        self.view, self.title = view, title
        eye, target = VIEWS[view]
        self.camera.set_world_poses_from_view(
            torch.tensor([eye], device=self.env.device) + self.env.scene.env_origins,
            torch.tensor([target], device=self.env.device) + self.env.scene.env_origins,
        )
        # Headless RTX is driven by sensor reads, not sim.render() alone.
        # Warm the actual render product without advancing physics.
        for _ in range(12):
            self.read_pixels()
        self.shots.append({"start_seconds": self.frames / self.fps, "view": view, "title": title})
        self.capture()
        self.active = True

    def read_pixels(self):
        """Fetch an actual RTX frame while leaving the simulation state unchanged."""
        self.env.sim.render()
        self.camera.update(self.env.physics_dt * self.stride, force_recompute=True)
        return self.camera.data.output["rgb"][0, :, :, :3].cpu().numpy()

    def capture(self):
        """Render and encode the current physics state, with a compact caption ribbon."""
        for _ in range(8):
            pixels = self.read_pixels()
            sample = pixels[::8, ::8]
            spatial_range = (sample.max(axis=(0, 1)) - sample.min(axis=(0, 1))).max()
            if spatial_range > 8:
                break
        else:
            Image.fromarray(pixels).save(self.directory / "blank_camera.png")
            diagnostics = {
                "view": self.view,
                "title": self.title,
                "encoded_frames": self.frames,
                "physics_step": self.env._sim_step_counter,
                "camera_position": self.camera.data.pos_w[:].tolist(),
                "camera_quaternion_opengl": self.camera.data.quat_w_opengl[:].tolist(),
                "rigid_states": {
                    name: asset.data.root_state_w[:].tolist() for name, asset in self.env.scene.rigid_objects.items()
                },
                "joint_positions": {
                    name: asset.data.joint_pos[:].tolist() for name, asset in self.env.scene.articulations.items()
                },
            }
            (self.directory / "blank_camera.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
            raise AssertionError(f"Camera {self.view!r} returned blank frames after warm-up")
        frame = Image.fromarray(pixels)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 76), fill=(14, 20, 28))
        draw.text((24, 10), f"{self.view.upper()}  |  {self.title}", font=self.font, fill=(242, 246, 250))
        draw.text(
            (24, 47),
            "Coffee grounds disposal  /  Force-assisted simulation",
            font=self.small_font,
            fill=(167, 199, 208),
        )
        if self.show_plan:
            draw.rounded_rectangle((28, 550, 735, 985), radius=18, fill=(14, 20, 28))
            draw.text((54, 574), "TASK OVERVIEW", font=self.font, fill=(225, 201, 161))
            lines = [
                "1   Raise dispenser; pull out the tray",
                "2   Open waste drawer; remove organic-bin lid",
                "3   Lift grounds container; tip and rock over bin",
                "4   Return container; slide tray into machine",
                "5   Replace bin lid; close waste drawer",
            ]
            for index, line in enumerate(lines):
                draw.text((54, 628 + index * 53), line, font=self.small_font, fill=(242, 246, 250))
            draw.text(
                (54, 926),
                "Goal: loose grounds in the bin, everything reseated.",
                font=self.small_font,
                fill=(167, 199, 208),
            )
        self.last_frame = frame.tobytes()
        self.encoder.stdin.write(self.last_frame)
        self.frames += 1

    def hold(self, seconds):
        """Hold the current shot on screen without extending the simulated episode."""
        assert self.last_frame is not None
        for _ in range(round(seconds * self.fps)):
            self.encoder.stdin.write(self.last_frame)
            self.frames += 1

    def finish(self, recording_valid, simulation_success):
        """Finalize a valid recording and report task success separately from smoke capture."""
        self.active = False
        self.encoder.stdin.close()
        code = self.encoder.wait(timeout=120)
        self.log.close()
        assert code == 0, f"Video encoder failed; see {self.directory / 'video_encoder.log'}"
        if recording_valid:
            self.partial.replace(self.path)
        report = {
            "file": self.path.name if recording_valid else self.partial.name,
            "fps": self.fps,
            "frames": self.frames,
            "duration_seconds": self.frames / self.fps,
            "resolution": [self.camera.cfg.width, self.camera.cfg.height],
            "recording_valid": recording_valid,
            "simulation_success": simulation_success,
            "shots": self.shots,
        }
        (self.directory / "video_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def make_speed_versions(directory):
    """Encode 2x/5x versions of the validated master without rerunning physics."""
    source = directory / "coffee_disposal_1x.mp4"
    for speed in (2, 5):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-vf",
                f"setpts=PTS/{speed},fps=30",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(directory / f"coffee_disposal_{speed}x.mp4"),
            ],
            check=True,
        )
