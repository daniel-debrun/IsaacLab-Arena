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

from isaaclab_arena_environments.pc_upgrade.cameras import EYE, TARGET

VIEWS = {
    "Overview": (EYE, TARGET),
    "Computer": ((0.96, -0.20, 1.32), (0.74, 0.40, 1.02)),
    "Top-down": ((0.35, -0.425, 1.37), (0.35, -0.424, 0.78)),
}


class VideoRecorderTerm(RecorderTerm):
    """Record at 30 Hz without changing the physics or policy control rates."""

    def record_post_physics_decimation_step(self):
        video = getattr(self._env, "pc_upgrade_video", None)
        if video is not None and video.active and self._env._sim_step_counter % video.stride == 0:
            video.capture()
        return None, None


class TaskVideo:
    """Stream labelled RGB frames to an MP4 encoder, with explicit shots and holds."""

    def __init__(self, env, directory: Path, fps=30):
        self.env = env
        self.directory = directory
        self.path = directory / "pc_upgrade_1x.mp4"
        self.partial = directory / "pc_upgrade_1x.partial.mp4"
        self.camera = env.scene["overview_cam"]
        self.fps = fps
        self.stride = round(1.0 / (env.physics_dt * fps))
        assert abs(self.stride * env.physics_dt - 1.0 / fps) < 1e-8
        self.active = False
        self.frames = 0
        self.shots = []
        self.view = "Overview"
        self.title = "PC upgrade"
        self.last_frame = None
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
        # Let exposure and temporal rendering settle after a cut; do not advance physics.
        for _ in range(12):
            self.env.sim.render()
        self.shots.append({"start_seconds": self.frames / self.fps, "view": view, "title": title})
        self.capture()
        self.active = True

    def capture(self):
        """Render and encode the current physics state, with a compact caption ribbon."""
        self.env.sim.render()
        self.camera.update(self.env.physics_dt * self.stride, force_recompute=True)
        pixels = self.camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
        frame = Image.fromarray(pixels)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 76), fill=(14, 20, 28))
        draw.text((24, 10), f"{self.view.upper()}  |  {self.title}", font=self.font, fill=(242, 246, 250))
        draw.text(
            (24, 47),
            "NVIDIA RTX 5090 upgrade  /  Force-assisted simulation",
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

    def finish(self, success):
        """Finalize the video and its shot list; publish the MP4 only after success."""
        self.active = False
        self.encoder.stdin.close()
        code = self.encoder.wait(timeout=120)
        self.log.close()
        assert code == 0, f"Video encoder failed; see {self.directory / 'video_encoder.log'}"
        if success:
            self.partial.replace(self.path)
        report = {
            "file": self.path.name if success else self.partial.name,
            "fps": self.fps,
            "frames": self.frames,
            "duration_seconds": self.frames / self.fps,
            "resolution": [self.camera.cfg.width, self.camera.cfg.height],
            "simulation_success": success,
            "shots": self.shots,
        }
        (self.directory / "video_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
