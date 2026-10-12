# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic wrapper around experiment_runner that logs RoboLab-style conditions during an Arena rollout.

Purely observational: success predicates, episode length and termination are untouched. It
  * adds a second contact sensor on the pick-up object filtered to the Robotiq finger links (gripper contact),
  * hooks TerminationManager.compute (state right after physics, before the reset) and records per step and env:
    object-in-container with RoboLab's open-top convex-hull rule (centroid of the object hull vertices vs the container hull
    planes with faces whose normal.z >= 0.7 dropped), gripper contact (|force| > 0.1 N, left finger only = RoboLab's
    contact_gripper, and any finger), object height/speed, Arena termination flags.
  * optional ablation --diag_robolab_physx: replaces Arena's stock PhysxCfg by RoboLab's solver settings
    (ccd, 32/1 iterations, bounce threshold 0.2, external forces every iteration off).  NOT a parity result.

Options: --diag_robolab_physx, --diag_replay=<npz prefix>:<run,run,...> (open-loop replay of RoboLab actions), --diag_out=<json>.
Usage (same arguments as experiment_runner.py; output goes to $DIAG_OUT, default <experiment_output_directory>/diag.json):
    python robolab_probe/high_diag.py --experiment_config X.yaml --experiment_output_directory OUT --enable_cameras ...
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys
import torch

DIAG = {"tasks": [], "rows": [], "meta": {}}
FILTER_FINGERS = [
    "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
    "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
]
FORCE_THRESHOLD = 0.1
OPTS = {"robolab_physx": False, "replay": None, "out": None}


def _pop_own_args():
    """Strip --diag_* options from sys.argv so experiment_runner's parser never sees them."""
    rest = []
    for a in sys.argv[1:]:
        if a == "--diag_robolab_physx":
            OPTS["robolab_physx"] = True
        elif a.startswith("--diag_replay="):
            OPTS["replay"] = a.split("=", 1)[1]
        elif a.startswith("--diag_out="):
            OPTS["out"] = a.split("=", 1)[1]
        else:
            rest.append(a)
    sys.argv[1:] = rest


def _patch_task_and_physics():
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.configclass import make_configclass

    from isaaclab.sensors import ContactSensorCfg

    orig_make_scene_cfg = PickAndPlaceTask.make_scene_cfg

    def make_scene_cfg(self):
        DIAG["tasks"].append(self)
        orig = orig_make_scene_cfg(self)
        if orig is None:
            return None
        diag_cfg = ContactSensorCfg(
            prim_path=self.pick_up_object.get_contact_sensor_prim_path(),
            filter_prim_paths_expr=list(FILTER_FINGERS),
        )
        base_cfg = self.pick_up_object.get_contact_sensor_cfg(contact_against_object=self.destination_location)
        scene_cfg_type = make_configclass(
            "SceneCfg",
            [(self.contact_sensor_cfg.name, type(base_cfg), base_cfg), (f"diag_gripper_contact_{self.pick_up_object.name}", type(diag_cfg), diag_cfg)],
        )
        return scene_cfg_type()

    PickAndPlaceTask.make_scene_cfg = make_scene_cfg

    if OPTS["robolab_physx"]:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        orig_compose = ArenaEnvBuilder.compose_manager_cfg

        def compose(self):
            env_cfg, kwargs = orig_compose(self)
            physx = env_cfg.sim.physics
            physx.enable_ccd = True
            physx.max_position_iteration_count = 32
            physx.max_velocity_iteration_count = 1
            physx.bounce_threshold_velocity = 0.2
            physx.enable_external_forces_every_iteration = False
            physx.gpu_temp_buffer_capacity = 2**30
            physx.gpu_heap_capacity = 2**30
            physx.gpu_collision_stack_size = 2**30
            print(f"[high_diag] applied RoboLab PhysX settings: {physx}", flush=True)
            return env_cfg, kwargs

        ArenaEnvBuilder.compose_manager_cfg = compose
        DIAG["meta"]["robolab_physx"] = True


def _mesh_points_in_body_frame(scene, key):
    """Mesh points of the rigid body prim of `key` (env 0) in its own frame (rot+trans removed, scale kept)."""
    from isaaclab_arena.environments import arena_world_scene_access as wa
    from pxr import Usd, UsdGeom

    prim = wa.get_representative_rigid_body_prims(scene, key)[0]
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    T_P_W = cache.GetLocalToWorldTransform(prim).RemoveScaleShear().GetInverse()
    pts = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        pt = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if pt is None or len(pt) == 0:
            continue
        m = cache.GetLocalToWorldTransform(p) * T_P_W
        arr = np.asarray(pt, dtype=np.float64)
        arr = np.c_[arr, np.ones(len(arr))] @ np.asarray(m).reshape(4, 4)  # USD row-vector convention
        pts.append(arr[:, :3])
    assert pts, f"no mesh points for {key}"
    return np.concatenate(pts, 0)


def _pose_e(T, scene):
    T = T.clone()
    T[:, :3] -= scene.env_origins
    return T.cpu().tolist()


class ReplayPolicy:
    """Open-loop replay of recorded RoboLab actions: env e, k-th episode plays demo e of run (runs[k]).

    Env var ARENA_DIAG_REPLAY="<prefix>:<run>,<run>,..." where files are <prefix><run>.npz with actions_<e>.
    """

    def __init__(self, spec, num_envs):
        prefix, runs = spec.rsplit(":", 1)
        self.runs = [np.load(f"{prefix}{r}.npz") for r in runs.split(",")]
        self.n = num_envs
        self.ep = np.zeros(num_envs, dtype=int)
        self.t = np.zeros(num_envs, dtype=int)
        self.task_description = None
        self.config = None

    def get_action(self, env, observation):
        acts = []
        for e in range(self.n):
            a = self.runs[min(self.ep[e], len(self.runs) - 1)][f"actions_{e}"]
            acts.append(a[min(self.t[e], len(a) - 1)])
            self.t[e] += 1
        return torch.tensor(np.stack(acts), dtype=torch.float32, device=env.unwrapped.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.ep[:] = 0
            self.t[:] = 0
        else:
            for e in env_ids.cpu().tolist():
                self.ep[e] += 1
                self.t[e] = 0

    def close(self):
        pass

    def set_task_description(self, d):
        self.task_description = d
        return d

    def has_length(self):
        return False

    def length(self):
        return None

    @property
    def is_remote(self):
        return False


def _install_hook(env):
    from scipy.spatial import ConvexHull

    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from isaaclab_arena.tasks.predicates.spatial import (
        contact_force_is_upward_support, object_bounds_center_over_destination)

    dev = env.device
    scene = env.scene
    per = {}
    tasks = list(DIAG["tasks"])
    DIAG["tasks"].clear()
    DIAG["rows"].clear()
    DIAG["meta"].clear()
    for task in tasks:
        obj, dest = task.pick_up_object.name, task.destination_location.name
        obj_pts = _mesh_points_in_body_frame(scene, obj)
        dest_pts = _mesh_points_in_body_frame(scene, dest)
        oh, dh = ConvexHull(obj_pts), ConvexHull(dest_pts)
        planes = torch.tensor(dh.equations, dtype=torch.float32, device=dev)
        per[obj] = dict(
            dest=dest,
            centroid_local=torch.tensor(obj_pts[oh.vertices].mean(0), dtype=torch.float32, device=dev),
            planes_open=planes[planes[:, 2] < 0.7],
            sensor=task.contact_sensor_cfg.name,
            fs=task.force_threshold,
            cone=task.support_cone_half_angle_rad,
            vthr=task.velocity_threshold,
        )
        DIAG["meta"][obj] = dict(dest=dest, obj_aabb_min=obj_pts.min(0).tolist(), obj_aabb_max=obj_pts.max(0).tolist(),
                                 dest_aabb_min=dest_pts.min(0).tolist(), dest_aabb_max=dest_pts.max(0).tolist())
    DIAG["meta"]["num_envs"] = env.num_envs
    DIAG["tag"] = "_".join(per)
    tm = env.termination_manager
    orig_compute = tm.compute
    state = {"step": 0}

    def compute(*a, **k):
        dones = orig_compute(*a, **k)
        with torch.no_grad():
            w = env.arena_world
            row = {"t": state["step"], "ep_len": env.episode_length_buf.cpu().tolist(), "done": dones.cpu().tolist(),
                   "terminated": tm.terminated.cpu().tolist(), "objs": {}}
            for obj, p in per.items():
                T_o, T_d = w.get_pose_w(obj), w.get_pose_w(p["dest"])
                c_w = quat_apply(T_o[:, 3:], p["centroid_local"].expand(env.num_envs, 3)) + T_o[:, :3]
                c_d = quat_apply_inverse(T_d[:, 3:], c_w - T_d[:, :3])
                in_hull = ((c_d @ p["planes_open"][:, :3].T + p["planes_open"][:, 3]).amax(-1) <= 0)
                over = object_bounds_center_over_destination(w.get_centroid_w(obj), T_d, w.get_aabb_in_local_frame(p["dest"]))
                fm = scene[f"diag_gripper_contact_{obj}"].data.force_matrix_w.torch
                f_grip = fm[:, 0].norm(dim=-1).amax(-1)  # max over the two finger filters
                sup = scene[p["sensor"]].data.force_matrix_w.torch[:, 0, 0, :]
                up = contact_force_is_upward_support(sup, p["fs"], p["cone"])
                speed = w.get_root_linear_velocity_w(obj).norm(dim=-1)
                row["objs"][obj] = {
                    "in_hull": in_hull.cpu().tolist(), "over": over.cpu().tolist(), "up": up.cpu().tolist(),
                    "sup_f": sup.norm(dim=-1).cpu().tolist(), "f_grip": f_grip.cpu().tolist(),
                    "speed": speed.cpu().tolist(), "z": (T_o[:, 2] - scene.env_origins[:, 2]).cpu().tolist(),
                    "c_dest": c_d.cpu().tolist(),
                }
            DIAG["rows"].append(row)
        state["step"] += 1
        if state["step"] % 150 == 0:
            _dump()
        return dones

    tm.compute = compute


def _dump():
    out = OPTS["out"]
    if not out:
        out_dir = sys.argv[sys.argv.index("--experiment_output_directory") + 1]
        out = os.path.join(out_dir, f"diag_{DIAG['tag']}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": DIAG["meta"], "rows": DIAG["rows"]}, f)
    print(f"[high_diag] wrote {out} rows={len(DIAG['rows'])}", flush=True)


def _patch_runner():
    from isaaclab_arena.evaluation import run_execution

    orig_rollout = run_execution.rollout_policy
    spec = OPTS["replay"]
    if spec:
        orig_build = run_execution._build_policy_from_cfg

        def build_policy(cfg):
            return ReplayPolicy(spec, cfg.environment_builder.num_envs)

        run_execution._build_policy_from_cfg = build_policy
        DIAG["meta"]["replay"] = spec

    def rollout(env, policy, num_steps, num_episodes):
        _install_hook(env.unwrapped)
        try:
            return orig_rollout(env, policy, num_steps, num_episodes)
        finally:
            _dump()

    run_execution.rollout_policy = rollout


def main():
    _pop_own_args()
    from isaaclab_arena.utils.isaaclab_utils import simulation_app as sa

    orig_enter = sa.SimulationAppContext.__enter__

    def enter(self):
        ret = orig_enter(self)
        _patch_task_and_physics()
        _patch_runner()
        return ret

    sa.SimulationAppContext.__enter__ = enter
    from isaaclab_arena.evaluation import experiment_runner

    experiment_runner.main()


if __name__ == "__main__":
    sys.exit(main())
