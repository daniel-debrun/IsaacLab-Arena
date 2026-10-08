# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-rate virtual grasp used only by the explicitly assisted demonstration."""

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)


def minimum_jerk(time: float, duration: float) -> tuple[float, float, float]:
    """Return position fraction and its time derivatives for a rest-to-rest motion."""
    assert duration > 0.0
    u = min(max(time / duration, 0.0), 1.0)
    return (
        10 * u**3 - 15 * u**4 + 6 * u**5,
        (30 * u**2 - 60 * u**3 + 30 * u**4) / duration,
        (60 * u - 180 * u**2 + 120 * u**3) / duration**2,
    )


def _limit(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    return vector * (maximum / vector.norm(dim=-1, keepdim=True).clamp_min(maximum))


class AssistedMotion(ActionTerm):
    """Apply bounded virtual-grasp wrenches before every physics step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.empty((env.num_envs, 0), device=env.device)
        self._motion = None
        self._panel_motion = None
        self._panel_target = None
        self._panel_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.telemetry = []

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        pass

    def reset(self, env_ids=None):
        self.release(env_ids)
        self._panel_active[slice(None) if env_ids is None else env_ids] = False
        if not self._panel_active.any():
            self._panel_motion = None
            self._panel_target = None

    def move(self, name, position, quaternion, duration, free_space=True):
        """Begin a smooth grasped motion to a world-frame pose."""
        if self._motion is not None and self._motion["name"] != name:
            self.release()
        asset = self._env.scene[name]
        initial_q = asset.data.root_quat_w[:].clone()
        rotation = axis_angle_from_quat(quat_mul(quaternion, quat_conjugate(initial_q)))
        self._motion = {
            "active": torch.ones(self._env.num_envs, dtype=torch.bool, device=self._env.device),
            "name": name,
            "asset": asset,
            "start": asset.data.root_pos_w[:].clone(),
            "quaternion": initial_q,
            "rotation": rotation,
            "target": position,
            "duration": duration,
            "elapsed": 0.0,
            "previous_position": asset.data.root_pos_w[:].clone(),
            "mass": asset.data.body_mass[:].reshape(-1, 1),
            "inertia": asset.data.body_inertia[:].reshape(-1, 3, 3),
        }
        self.telemetry.append({
            "asset": name,
            "free_space": free_space,
            "max_tracking_error_m": 0.0,
            "max_orientation_error_rad": 0.0,
            "max_step_displacement_m": 0.0,
            "max_speed_m_s": 0.0,
        })

    def release(self, env_ids=None):
        """Remove the virtual grasp so gravity and contacts determine the motion."""
        if self._motion is not None:
            self._motion["asset"].permanent_wrench_composer.reset(env_ids=env_ids)
            self._motion["active"][slice(None) if env_ids is None else env_ids] = False
            if not self._motion["active"].any():
                self._motion = None

    def move_panel(self, angle, duration):
        """Move the demonstration's panel position drive smoothly to an angle."""
        initial = self._env.scene["pc_chassis"].data.joint_pos[:].clone()
        self._panel_motion = (initial, angle, duration, 0.0)
        self._panel_active[:] = True

    def apply_actions(self):
        dt = self._env.physics_dt
        if self._panel_motion is not None:
            initial, angle, duration, elapsed = self._panel_motion
            elapsed += dt
            fraction, _, _ = minimum_jerk(elapsed, duration)
            self._panel_target = initial * (1.0 - fraction) + angle * fraction
            self._panel_motion = (initial, angle, duration, elapsed)
        if self._panel_target is not None:
            panel_ids = self._panel_active.nonzero(as_tuple=False).flatten()
            self._env.scene["pc_chassis"].set_joint_position_target(self._panel_target[panel_ids], env_ids=panel_ids)
        motion = self._motion
        if motion is None:
            return
        asset = motion["asset"]
        fraction, rate, acceleration = minimum_jerk(motion["elapsed"], motion["duration"])
        displacement = motion["target"] - motion["start"]
        target = motion["start"] + fraction * displacement
        desired_velocity = rate * displacement
        position = asset.data.root_pos_w[:]
        velocity = asset.data.root_link_lin_vel_w[:]
        error = target - position
        linear_acceleration = _limit(
            225.0 * error + 30.0 * (desired_velocity - velocity) + acceleration * displacement,
            15.0,
        )
        gravity = position.new_tensor(self._env.cfg.sim.gravity)
        force = motion["mass"] * (linear_acceleration - gravity)
        rotation = motion["rotation"]
        delta_q = quat_from_angle_axis(fraction * rotation.norm(dim=-1), rotation)
        target_q = quat_mul(delta_q, motion["quaternion"])
        current_q = asset.data.root_quat_w[:]
        orientation_error = axis_angle_from_quat(quat_mul(target_q, quat_conjugate(current_q)))
        angular_velocity = asset.data.root_ang_vel_w[:]
        angular_acceleration = _limit(
            100.0 * orientation_error + 20.0 * (rate * rotation - angular_velocity) + acceleration * rotation,
            15.0,
        )
        angular_acceleration_b = quat_apply_inverse(current_q, angular_acceleration)
        torque_b = torch.bmm(motion["inertia"], angular_acceleration_b.unsqueeze(-1)).squeeze(-1)
        torque = _limit(quat_apply(current_q, torque_b), 2.0)
        active_ids = motion["active"].nonzero(as_tuple=False).flatten()
        asset.permanent_wrench_composer.reset(env_ids=active_ids)
        asset.permanent_wrench_composer.add_forces_and_torques_index(
            forces=force[active_ids, None, :],
            torques=torque[active_ids, None, :],
            env_ids=active_ids,
            is_global=True,
        )
        metrics = self.telemetry[-1]
        metrics["max_tracking_error_m"] = max(
            metrics["max_tracking_error_m"], error[active_ids].norm(dim=-1).max().item()
        )
        metrics["max_orientation_error_rad"] = max(
            metrics["max_orientation_error_rad"], orientation_error[active_ids].norm(dim=-1).max().item()
        )
        metrics["max_step_displacement_m"] = max(
            metrics["max_step_displacement_m"],
            (position - motion["previous_position"])[active_ids].norm(dim=-1).max().item(),
        )
        metrics["max_speed_m_s"] = max(metrics["max_speed_m_s"], velocity[active_ids].norm(dim=-1).max().item())
        motion["previous_position"] = position.clone()
        motion["elapsed"] += dt


@configclass
class AssistedMotionCfg(ActionTermCfg):
    """Enable the controller only when constructing the assisted demonstration."""

    class_type: type = AssistedMotion
    asset_name: str = "rtx_5090"
