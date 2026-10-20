# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image encoding utilities."""

import base64
import io
import numpy as np

from PIL import Image


def encode_image(image: np.ndarray, max_width_or_height: int = 384) -> str:
    """Encode an RGB or RGBA image as JPEG, preserving aspect ratio without upscaling.

    Args:
        image: HWC uint8 pixels; the alpha channel, if present, is discarded.
        max_width_or_height: Maximum encoded image width or height, in pixels. The 384 default
            comes from the original Astra VLM policy's downsampling setting; JPEG does not require it.

    Returns:
        A JPEG data URL for an image-capable chat-completion endpoint.
    """
    assert (
        isinstance(max_width_or_height, int) and max_width_or_height > 0
    ), "max_width_or_height must be a positive integer"
    assert image.ndim == 3 and image.shape[-1] in (3, 4), f"Unexpected image shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 pixels, got {image.dtype}"
    frame = Image.fromarray(image[..., :3])
    frame.thumbnail((max_width_or_height, max_width_or_height))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
