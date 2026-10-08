# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wrist and overview cameras for inspection and visual policies."""

import torch

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_arena.embodiments.franka.franka import FrankaCameraCfg

EYE = (1.55, -1.6, 1.90)
TARGET = (0.43, 0.03, 0.96)


@configclass
class PCUpgradeCameraCfg(FrankaCameraCfg):
    overview_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        height=900,
        width=1200,
        data_types=["rgb"],
        update_period=0.0,
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=32.0, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(
            pos=EYE,
            rot=tuple(
                quat_from_matrix(create_rotation_matrix_from_view(torch.tensor([EYE]), torch.tensor([TARGET])))[
                    0
                ].tolist()
            ),
            convention="opengl",
        ),
    )
