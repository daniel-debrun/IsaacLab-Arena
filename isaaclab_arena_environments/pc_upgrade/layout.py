# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene dimensions in metres, in the local environment frame E."""

TABLE_Z = 0.75
PC_POSITION = (0.74, 0.36, TABLE_Z)
SLOT_IN_PC = (-0.060, 0.045, 0.205)
# Fan face points down; the gold edge connector points toward the motherboard (+Y).
SLOT_QUATERNION = (1.0, 0.0, 0.0, 0.0)
BIN_POSITION = (0.20, 0.40, TABLE_Z)
CARD_POSITIONS = {
    "rtx_5090": (0.48, -0.32, TABLE_Z + 0.025),
    "vortex_vx880": (0.15, -0.53, TABLE_Z + 0.025),
    "cinder_c700": (0.53, -0.53, TABLE_Z + 0.025),
}
