# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted approach policy and task for the avocado settling demonstration.

Run with experiment_runner.py and the avocado_settling_experiment.yaml configuration.
The gripper approaches a point 20 cm above the recorded avocado rest position; it does not grasp.
"""

import torch
from dataclasses import dataclass
from functools import partial

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state, objects_settled
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.terms.events import write_root_velocity_to_sim

HOVER_HEIGHT_M = 0.20


def launch_avocado(env, env_ids, drop_height_m, sideways_speed_m_s, spin_rad_s):
    """Lift the relation-placed avocado and assign its drop velocity on each reset."""
    avocado = env.scene["avocado"]
    T_W_O = avocado.data.root_pose_w.torch[env_ids].clone()
    T_W_O[:, 2] += drop_height_m
    avocado.write_root_pose_to_sim_index(root_pose=T_W_O, env_ids=env_ids)
    velocity_W = torch.zeros((len(env_ids), 6), device=env.device)
    velocity_W[:, 1] = sideways_speed_m_s
    velocity_W[:, 2] = -0.05
    velocity_W[:, 3] = spin_rad_s
    write_root_velocity_to_sim(env, avocado, velocity_W, env_ids)


def hover_target_w(env):
    """Return the fixed world-frame target above the avocado's recorded rest position."""
    t_W_O, _ = get_object_initial_rest_state(env, "avocado")
    t_W_target = t_W_O.clone()
    t_W_target[:, 2] += HOVER_HEIGHT_M
    return t_W_target


def gripper_at_hover_target(env):
    """Return whether the gripper is within 2 cm of the hover target."""
    t_W_G = env.arena_world.get_frame_position_w("ee_frame", "end_effector")
    return torch.linalg.vector_norm(hover_target_w(env) - t_W_G, dim=-1) < 0.02


class AvocadoSettlingTask(NoTask):
    """Require initial rest followed by a sustained gripper arrival above the avocado."""

    def __init__(self, episode_length_s, settled_steps):
        super().__init__(episode_length_s, "Wait for the avocado to settle, then move above it.")
        self.settled_steps = settled_steps

    def get_termination_cfg(self):
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            failures={
                "avocado_fell_off_table": TerminationTermCfg(
                    func=mdp.root_height_below_minimum,
                    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("avocado")},
                ),
            },
            success=[
                CompletionCriteria(
                    name="settle_then_approach",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(
                            predicate=partial(objects_settled, object_names=["avocado"]),
                            required_steps=self.settled_steps,
                        ),
                        TrueForConsecutiveStepsCfg(predicate=gripper_at_hover_target, required_steps=10),
                    ],
                )
            ],
        )


@dataclass
class AvocadoHoverPolicyCfg(PolicyCfg):
    """Configure the speed of the scripted gripper approach."""

    speed_m_s: float = 0.10

    def __post_init__(self):
        assert self.speed_m_s > 0


@register_policy
class AvocadoHoverPolicy(PolicyBase[AvocadoHoverPolicyCfg]):
    """Move the Franka tool toward the hover target with bounded relative IK commands."""

    name = "avocado_hover"

    def __init__(self, config):
        super().__init__(config)
        self._reported_start = False

    def reset(self, env_ids=None):
        self._reported_start = False

    def get_action(self, env, observation):
        arena_env = env.unwrapped
        pending = arena_env.progress_tracker.get_initial_object_settling_pending_mask()
        assert not bool(
            (pending & arena_env.active_episode_mask).any()
        ), "Enable policy.wait_for_objects_to_settle for this demonstration."
        if not self._reported_start:
            print(f"[avocado demo] Settled; policy starts at episode step {arena_env.episode_length_buf.tolist()}.")
            print(f"[avocado demo] Hover target in world frame: {hover_target_w(arena_env).tolist()}")
            self._reported_start = True
        t_W_G = arena_env.arena_world.get_frame_position_w("ee_frame", "end_effector")
        delta_W = hover_target_w(arena_env) - t_W_G
        distance = torch.linalg.vector_norm(delta_W, dim=-1, keepdim=True)
        if int(arena_env.episode_length_buf[0]) % 60 == 0:
            print(f"[avocado demo] Tool position W: {t_W_G.tolist()}, target error: {distance.tolist()}")
        delta_W *= (self.config.speed_m_s * arena_env.step_dt / distance.clamp_min(1e-6)).clamp(max=1.0)
        # Relative IK translation commands are expressed in robot base frame B.
        q_W_B = arena_env.scene["robot"].data.root_quat_w.torch
        delta_B = quat_apply_inverse(q_W_B, delta_W)
        actions = torch.zeros(env.action_space.shape, device=arena_env.device)
        actions[:, :3] = delta_B / 0.5  # FrankaIKActionCfg translation scale.
        actions[:, -1] = 1.0  # Keep the gripper open.
        return actions
