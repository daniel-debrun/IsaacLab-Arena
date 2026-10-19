# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image preparation utilities."""

import numpy as np
import torch


def resize_rgb_image(image: torch.Tensor, image_max_edge: int) -> np.ndarray:
    """Copy an RGB or RGBA image as RGB, resizing without upscaling.

    Args:
        image: HWC uint8 image.
        image_max_edge: Maximum output width or height, preserving aspect ratio.

    Returns:
        Independent HWC uint8 RGB array.
    """
    assert isinstance(image_max_edge, int) and image_max_edge > 0, "image_max_edge must be a positive integer"
    assert image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0, "Expected a nonempty HWC image"
    assert image.dtype == torch.uint8 and image.shape[-1] in (3, 4), "Expected unnormalized uint8 RGB or RGBA"
    source_height, source_width = image.shape[:2]
    ratio = min(1.0, image_max_edge / max(source_height, source_width))
    height, width = max(1, round(source_height * ratio)), max(1, round(source_width * ratio))
    rgb = image[..., :3]
    if (height, width) != (source_height, source_width):
        pixels = rgb.permute(2, 0, 1).unsqueeze(0).float()
        pixels = torch.nn.functional.interpolate(
            pixels, size=(height, width), mode="bilinear", align_corners=False, antialias=True
        )
        rgb = pixels[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
    return rgb.cpu().numpy().copy()
