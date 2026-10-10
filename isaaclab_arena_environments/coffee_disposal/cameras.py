# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Overview and wrist RGB cameras for the coffee-disposal kitchen."""

import torch

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_arena.embodiments.franka.franka import FrankaCameraCfg

EYE = (2.25, -2.8, 2.25)
TARGET = (0.15, -0.08, 0.80)
VIEWS = {
    "Overview": (EYE, TARGET),
    "Machine": ((1.30, -0.90, 1.65), (0.66, 0.10, 1.09)),
    "Waste drawer": ((0.25, -1.65, 1.48), (-0.31, -0.62, 0.44)),
    "Grounds close-up": ((0.02, -1.17, 1.36), (-0.18, -0.75, 0.66)),
    "Bin interior": ((-0.18, -0.78, 1.10), (-0.18, -0.75, 0.30)),
    "Top-down": ((0.65, -0.11, 1.80), (0.65, -0.10, 0.94)),
}


@configclass
class CoffeeDisposalCameraCfg(FrankaCameraCfg):
    overview_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        height=1080,
        width=1440,
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
