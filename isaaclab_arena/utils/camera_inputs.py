# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare RGB inputs with calibration in a caller-selected reference frame."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.utils.image import resize_rgb_image


@dataclass(frozen=True)
class CameraInput:
    """Store one camera image and optional calibration in a reference frame."""

    rgb: np.ndarray
    """Copied HWC uint8 RGB image, resized without cropping, padding, or upscaling."""

    intrinsics: np.ndarray | None = None
    """Pinhole matrix scaled to rgb's actual width and height, shaped (3, 3)."""

    T_R_C: np.ndarray | None = None
    """Optical camera C in reference R, meters and XYZW; C uses +X right, +Y down, +Z forward."""


def enable_camera_pose_updates(camera_cfg: Any, camera_names: tuple[str, ...]) -> None:
    """Enable per-step calibration before building the environment.

    Args:
        camera_cfg: Arena camera configuration to update in place.
        camera_names: Camera sensor field names in camera_cfg, independent of observation modality.
    """
    for camera_name in camera_names:
        cfg = getattr(camera_cfg, camera_name)
        cfg.update_latest_camera_pose = True
        cfg.update_period = 0.0


def extract_camera_inputs_in_reference_frame(
    env: Any,
    observation: dict,
    env_id: int,
    camera_names_by_obs_key: Mapping[str, str],
    *,
    T_W_R: torch.Tensor | None = None,
    image_max_edge: int = 384,
    include_calibration: bool = True,
) -> dict[str, CameraInput]:
    """Copy camera images with optional optical camera poses in reference frame R.

    Args:
        env: Arena environment, optionally gym-wrapped; do not step between observation and extraction.
        observation: Current observations with unnormalized uint8 camera_obs terms.
        env_id: Environment whose images and calibration to extract.
        camera_names_by_obs_key: Mapping from RGB observation keys to scene camera sensor names.
            Keys may use any naming convention; values identify the cameras supplying calibration.
        T_W_R: Reference R in simulation world W for each environment, shaped (N, 7),
            meters and unit XYZW quaternions. Required with calibration. Pass a robot
            root pose for robot-relative calibration, or identity poses for world-relative
            calibration. Must correspond to the same simulation step as observation.
        image_max_edge: Maximum resized width or height, preserving aspect ratio.
        include_calibration: Require per-step camera pose updates enabled before environment construction.

    Returns:
        Inputs keyed by observation name. Calibration uses camera data from the
        same update as the images. Intrinsics alone do not determine scene depth.
    """
    arena_env = env.unwrapped
    assert 0 <= env_id < arena_env.num_envs, "env_id is out of range"
    assert isinstance(image_max_edge, int) and image_max_edge > 0, "image_max_edge must be a positive integer"
    if include_calibration:
        assert T_W_R is not None, "T_W_R is required for calibration"
        _validate_reference_poses(T_W_R, arena_env.num_envs)

    result = {}
    for key, camera_name in camera_names_by_obs_key.items():
        images = observation["camera_obs"][key]
        assert images.ndim == 4 and images.shape[0] == arena_env.num_envs, "Expected batched HWC camera images"
        image = images[env_id].detach()
        rgb = resize_rgb_image(image, image_max_edge)
        intrinsics, T_R_C = None, None
        if include_calibration:
            camera = arena_env.scene[camera_name]
            intrinsics, T_R_C = _extract_camera_calibration_in_reference_frame(
                camera, env_id, T_W_R, image.shape[:2], rgb.shape[:2]
            )
        result[key] = CameraInput(rgb=rgb, intrinsics=intrinsics, T_R_C=T_R_C)
    return result


def _validate_reference_poses(T_W_R: torch.Tensor, num_envs: int) -> None:
    """Validate a batch of reference poses in simulation world coordinates.

    Args:
        T_W_R: Reference R in world W, shaped (num_envs, 7), meters and XYZW.
        num_envs: Expected number of reference poses.
    """
    assert T_W_R.shape == (num_envs, 7), "Expected one reference pose per environment"
    assert torch.isfinite(T_W_R).all(), "Reference poses must be finite"
    quaternion_norm = torch.linalg.vector_norm(T_W_R[:, 3:], dim=-1)
    assert torch.allclose(
        quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-4
    ), "Expected unit XYZW reference quaternions"


def _extract_camera_calibration_in_reference_frame(
    camera: Any,
    env_id: int,
    T_W_R: torch.Tensor,
    source_shape: tuple[int, int],
    resized_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Copy resized intrinsics and the optical camera pose in reference frame R.

    Args:
        camera: Camera sensor with per-step pose updates enabled.
        env_id: Environment whose calibration to extract.
        T_W_R: Batched reference poses in world W, meters and XYZW.
        source_shape: Original image (height, width).
        resized_shape: Output image (height, width).

    Returns:
        Rescaled pinhole matrix and T_R_C pose, in meters and XYZW.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    assert (
        camera.cfg.update_latest_camera_pose and camera.cfg.update_period == 0
    ), "Call enable_camera_pose_updates(camera_cfg, camera_names) before building the environment"
    data = camera.data
    assert tuple(data.image_shape) == tuple(source_shape), "Image dimensions do not match camera calibration"
    source_height, source_width = source_shape
    height, width = resized_shape
    intrinsics = data.intrinsic_matrices.torch[env_id].detach().cpu().numpy().copy()
    intrinsics[0, :] *= width / source_width
    intrinsics[1, :] *= height / source_height
    t_W_C = data.pos_w.torch[env_id : env_id + 1]  # Camera C's origin in world W, meters; retain the batch dimension.
    q_W_C = data.quat_w_ros.torch[env_id : env_id + 1]  # Optical C-to-W rotation, XYZW.
    # Select reference R's world pose for this environment, matching the camera tensor's device and dtype.
    T_W_R_env = T_W_R[env_id : env_id + 1].to(device=t_W_C.device, dtype=t_W_C.dtype)
    # Express camera C in reference R: T_R_C = inverse(T_W_R) * T_W_C.
    t_R_C, q_R_C = subtract_frame_transforms(T_W_R_env[:, :3], T_W_R_env[:, 3:], t_W_C, q_W_C)
    # Return an independent CPU pose [x, y, z, qx, qy, qz, qw], without the batch dimension.
    T_R_C = torch.cat((t_R_C, q_R_C), dim=-1)[0].detach().cpu().numpy().copy()
    return intrinsics, T_R_C
