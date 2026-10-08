# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched geometric checks; poses use scalar-last quaternions, in the target frame."""

import torch

CARD_HALF_EXTENTS = (0.153, 0.071, 0.0275)
CARD_BOUNDS_CENTER = (0.0, 0.0, 0.0065)
BIN_HALF_WIDTH = (0.205, 0.165)
BIN_HEIGHT = 0.30


def seated_card(t_S_G, q_S_G, velocity):
    """Return whether GPU G is stationary and aligned with slot S (metres, xyzw)."""
    close = torch.abs(t_S_G).le(t_S_G.new_tensor([0.012, 0.012, 0.008])).all(dim=-1)
    aligned = torch.abs(q_S_G[:, 3]) > 0.9962  # Within ten degrees, including q == -q.
    return close & aligned & settled(velocity)


def settled(velocity):
    """Return whether linear and angular speed are below 3 cm/s and 0.15 rad/s."""
    return (torch.linalg.vector_norm(velocity[:, :3], dim=-1) < 0.03) & (
        torch.linalg.vector_norm(velocity[:, 3:], dim=-1) < 0.15
    )


def contained_card(t_B_G, q_B_G, velocity):
    """Return whether the entire oriented GPU fits inside bin B and is stationary."""
    x, y, z, w = q_B_G.unbind(-1)
    row_x = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1)
    row_y = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1)
    row_z = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1)
    rotation = torch.stack((row_x, row_y, row_z), dim=1)
    extent = torch.abs(rotation) @ t_B_G.new_tensor(CARD_HALF_EXTENTS)
    t_B_G = t_B_G + rotation @ t_B_G.new_tensor(CARD_BOUNDS_CENTER)
    inside_xy = (torch.abs(t_B_G[:, :2]) + extent[:, :2] < t_B_G.new_tensor(BIN_HALF_WIDTH)).all(-1)
    above_floor = t_B_G[:, 2] - extent[:, 2] >= 0.003
    below_rim = t_B_G[:, 2] + extent[:, 2] < BIN_HEIGHT - 0.005
    return inside_xy & above_floor & below_rim & settled(velocity)


def hand_clear_of_card(t_G_H):
    """Require the hand H to be at least 8 cm outside GPU G's visual bounding box."""
    relative = t_G_H - t_G_H.new_tensor(CARD_BOUNDS_CENTER)
    gap = (relative.abs() - t_G_H.new_tensor(CARD_HALF_EXTENTS)).clamp_min(0)
    return torch.linalg.vector_norm(gap, dim=-1) > 0.08
