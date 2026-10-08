# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end physics validation, isolated from pytest's SimulationApp lifecycle."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants


@pytest.mark.with_subprocess
def test_pc_upgrade_assisted_sequence_in_two_environments(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        TestConstants.python_path,
        str(root / "isaaclab_arena_examples/pc_upgrade/demo.py"),
        "--num_envs",
        "2",
        "--output",
        str(tmp_path),
    ]
    log_path = tmp_path / "simulation.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, log_path.read_text()[-6000:]
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["success"] == [True, True]
    assert report["reset_verified"]
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 2
    assert all(episode["progress"]["all_complete"] for episode in episodes)
