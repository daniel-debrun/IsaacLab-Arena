# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared RGB image preparation."""

import numpy as np
import torch

import pytest

from isaaclab_arena.utils.image import resize_rgb_image


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("image_max_edge, expected_shape", [(4, (2, 4, 3)), (20, (5, 9, 3))])
def test_resize_rgb_image(channels, image_max_edge, expected_shape):
    image = torch.full((5, 9, channels), 255, dtype=torch.uint8)
    image[..., :3] = torch.tensor([25, 80, 160], dtype=torch.uint8)

    rgb = resize_rgb_image(image, image_max_edge)

    assert rgb.shape == expected_shape
    assert rgb.dtype == np.uint8
    image.zero_()
    np.testing.assert_array_equal(rgb, np.broadcast_to([25, 80, 160], expected_shape))


@pytest.mark.parametrize("image_max_edge", [0, -1, 1.5])
def test_resize_rgb_image_invalid_max_edge(image_max_edge):
    with pytest.raises(AssertionError, match="positive integer"):
        resize_rgb_image(torch.zeros((5, 9, 3), dtype=torch.uint8), image_max_edge)


@pytest.mark.parametrize("channels, dtype", [(1, torch.uint8), (3, torch.float32)])
def test_resize_rgb_image_rejects_non_rgb(channels, dtype):
    with pytest.raises(AssertionError, match="uint8 RGB or RGBA"):
        resize_rgb_image(torch.zeros((5, 9, channels), dtype=dtype), 4)
