# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena_environments.pc_upgrade.geometry import contained_card, seated_card


def test_seating_rejects_wrong_orientation_motion_and_height():
    position = torch.tensor([[0.0, 0.0, 0.0]]).repeat(5, 1)
    position[1, 2] = 0.03
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(5, 1)
    quaternion[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    quaternion[4] *= -1  # Same rotation in quaternion double cover.
    velocity = torch.zeros(5, 6)
    velocity[3, 0] = 0.2
    assert seated_card(position, quaternion, velocity).tolist() == [True, False, False, False, True]


def test_bin_requires_whole_card_below_rim_and_inside_walls():
    position = torch.tensor([[0.0, 0.0, 0.10], [0.12, 0.0, 0.10], [0.0, 0.0, 0.30]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    velocity = torch.zeros(3, 6)
    assert contained_card(position, quaternion, velocity).tolist() == [True, False, False]


def test_rotated_card_cannot_protrude_through_bin_wall():
    position = torch.tensor([[0.0, 0.03, 0.10]])
    quaternion = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])
    assert not contained_card(position, quaternion, torch.zeros(1, 6)).item()


def test_bin_rejects_card_still_falling_or_spinning():
    position = torch.tensor([[0.0, 0.0, 0.10]]).repeat(2, 1)
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    velocity = torch.zeros(2, 6)
    velocity[0, 2] = -0.2
    velocity[1, 5] = 0.5
    assert not contained_card(position, quaternion, velocity).any()


def test_bin_checks_fan_height_not_only_collision_box():
    position = torch.tensor([[0.0, 0.0, 0.265]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert not contained_card(position, quaternion, torch.zeros(1, 6)).item()


def test_release_rejects_hand_at_long_card_tip():
    from isaaclab_arena_environments.pc_upgrade.geometry import hand_clear_of_card

    hand = torch.tensor([[0.14, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.16, 0.0]])
    assert hand_clear_of_card(hand).tolist() == [False, True, True]
