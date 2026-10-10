# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stationary camera cuts and terminal holds must contain rendered scene pixels."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_coffee_camera_cuts_without_physics(simulation_app):
    import json
    import numpy as np
    import torch
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    from PIL import Image

    from isaaclab_arena_environments.coffee_disposal.cameras import VIEWS
    from isaaclab_arena_examples.coffee_disposal.demo import Demo, record_demo_terminal
    from isaaclab_arena_examples.coffee_disposal.video import TaskVideo

    class FrameSink:
        def __init__(self):
            self.last = None
            self.count = 0

        def write(self, frame):
            self.last = frame
            self.count += 1

    scenes = json.loads((Path(__file__).with_name("data") / "coffee_disposal_camera_scenes.json").read_text())
    artifacts = Path("/tmp/coffee-camera-cuts-validation")
    artifacts.mkdir(exist_ok=True)
    diagnostics = []
    with TemporaryDirectory(prefix="coffee-video-test-") as directory:
        args = SimpleNamespace(
            output=Path(directory),
            enable_cameras=True,
            video=False,
            capture=False,
            num_envs=2,
            device="cuda:0",
            real_time=False,
        )
        demo = Demo(args)
        sink = FrameSink()
        video = None
        try:
            with patch("isaaclab_arena_examples.coffee_disposal.video.subprocess.Popen") as encoder:
                encoder.return_value = SimpleNamespace(stdin=sink)
                video = TaskVideo(demo.base, args.output)
            demo.video = video
            base = demo.base
            initial_counter = base._sim_step_counter

            def assert_scene_pixels(label):
                pixels = np.frombuffer(sink.last, dtype=np.uint8).reshape(
                    video.camera.cfg.height, video.camera.cfg.width, 3
                )
                # The caption can disguise a blank camera image, so exclude its ribbon.
                scene_pixels = pixels[90:, :, :]
                spread = float(np.ptp(scene_pixels, axis=(0, 1)).max())
                deviation = float(scene_pixels.astype(np.float32).std(axis=(0, 1)).max())
                diagnostics.append({"shot": label, "range": spread, "std": deviation})
                Image.fromarray(pixels).save(artifacts / f"{label}.png")
                assert spread > 32 and deviation > 5, diagnostics[-1]
                assert base._sim_step_counter == initial_counter, "Camera warming and holds must not advance physics"

            for scene in scenes:
                for name, rows in scene.items():
                    if name == "stage":
                        continue
                    if name.endswith("_joints"):
                        asset = base.scene[name.removesuffix("_joints")]
                        joints = torch.tensor(rows, device=base.device)
                        asset.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                        asset.set_joint_position_target(joints)
                    else:
                        asset = base.scene[name]
                        state = asset.data.root_state_w[:].clone()
                        state[:, :7] = torch.tensor(rows, device=base.device)
                        state[:, :3] += base.scene.env_origins
                        state[:, 7:] = 0.0
                        asset.write_root_state_to_sim(state)
                # Publish restored physics state once before measuring stationary cuts.
                demo.env.step(demo.action)
                initial_counter = base._sim_step_counter
                states_before = {
                    name: base.scene[name].data.root_state_w[:].clone()
                    for name in scene
                    if name != "stage" and not name.endswith("_joints")
                }
                for view in VIEWS:
                    title = scene["stage"]
                    if scene["stage"] == "08_bin_contents" and view == "Bin interior":
                        title = "Loose grounds retained inside the organic bin"
                    video.shot(view, title)
                    assert_scene_pixels(f"{scene['stage']}_{view.replace(' ', '_')}")
                    if scene["stage"] == "08_bin_contents" and view == "Bin interior":
                        Image.frombytes("RGB", (video.camera.cfg.width, video.camera.cfg.height), sink.last).save(
                            "/tmp/coffee_corrected_bin.png"
                        )
                for name, before in states_before.items():
                    assert torch.equal(base.scene[name].data.root_state_w[:], before)

            base.progress_tracker._task_success.fill_(True)
            for runner in base.progress_tracker.runners:
                for name in runner.sequence_complete:
                    runner.sequence_complete[name].fill_(True)
            before_frames = video.frames
            record_demo_terminal(base, 0)
            assert video.frames - before_frames == 1 + 4 * video.fps
            assert sink.count == video.frames
            assert not video.active
            assert demo.evidence[-1]["stage"] == "11_success"
            assert_scene_pixels("terminal_overview_hold")
            Image.frombytes("RGB", (video.camera.cfg.width, video.camera.cfg.height), sink.last).save(
                "/tmp/coffee_corrected_complete.png"
            )
            (artifacts / "validation.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
            print("COFFEE_CAMERA_CUTS " + json.dumps(diagnostics), flush=True)
            return True
        finally:
            if video is not None:
                video.log.close()
            demo.env.close()


@pytest.mark.with_cameras
def test_coffee_camera_cuts_without_physics():
    assert run_function_with_persistent_simulation_app(_test_coffee_camera_cuts_without_physics, enable_cameras=True)
