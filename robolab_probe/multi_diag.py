# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic wrapper around experiment_runner for composite PickAndPlace tasks (observational only).

For every PickAndPlaceTask subtask (pick-up object -> destination) it logs, per control step and env, right after
physics and before the reset:
  * RoboLab-style object_in_container (centroid of the object's hull vertices inside the OPEN-TOP convex hull of the
    destination), finger contact (|force| > 0.1 N, left/right inner finger of the Robotiq gripper), speed, height,
    centroid in the destination frame,
  * the components of Arena's object_on_destination: bounds centre over destination, filtered support force from
    the destination (norm + upward-cone), mean speed, plus an UNFILTERED net contact force on the object (support
    from anything, e.g. fruit in the bin),
  * which termination terms are true (success, time_out, object_dropped_subtask_k...).
Success predicates, episode length and terminations are untouched.

Usage (same arguments as experiment_runner.py): python robolab_probe/multi_diag.py --experiment_config X.yaml ...
Env DIAG_OUT overrides the output json (default <experiment_output_directory>/multi_diag.json).
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys
import torch

DIAG = {"tasks": {}, "rows": [], "meta": {}}
FINGERS = [
    "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
    "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
]


def _patch_task():
    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.configclass import make_configclass

    orig = PickAndPlaceTask.make_scene_cfg

    def make_scene_cfg(self):
        DIAG["tasks"][self.pick_up_object.name] = self
        base = orig(self)
        if base is None:
            return None
        name = self.pick_up_object.name
        prim = self.pick_up_object.get_contact_sensor_prim_path()
        fing = ContactSensorCfg(prim_path=prim, filter_prim_paths_expr=list(FINGERS))
        anyc = ContactSensorCfg(prim_path=prim, filter_prim_paths_expr=[])
        base_cfg = self.pick_up_object.get_contact_sensor_cfg(contact_against_object=self.destination_location)
        return make_configclass(
            "SceneCfg",
            [
                (self.contact_sensor_cfg.name, type(base_cfg), base_cfg),
                (f"diag_finger_{name}", type(fing), fing),
                (f"diag_any_{name}", type(anyc), anyc),
            ],
        )()

    PickAndPlaceTask.make_scene_cfg = make_scene_cfg


def _mesh_points_in_body_frame(scene, key):
    from pxr import Usd, UsdGeom

    from isaaclab_arena.environments import arena_world_scene_access as wa

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
        arr = np.c_[arr, np.ones(len(arr))] @ np.asarray(m).reshape(4, 4)
        pts.append(arr[:, :3])
    assert pts, f"no mesh points for {key}"
    return np.concatenate(pts, 0)


def _install_hook(env):
    from scipy.spatial import ConvexHull

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_arena.tasks.predicates.spatial import object_bounds_center_over_destination

    scene, dev = env.scene, env.device
    info = {}
    dest_cache = {}
    for obj, task in DIAG["tasks"].items():
        dest = task.destination_location.name
        pts = _mesh_points_in_body_frame(scene, obj)
        oh = ConvexHull(pts)
        if dest not in dest_cache:
            dpts = _mesh_points_in_body_frame(scene, dest)
            dh = ConvexHull(dpts)
            planes = torch.tensor(dh.equations, dtype=torch.float32, device=dev)
            dest_cache[dest] = planes[planes[:, 2] < 0.7]
        info[obj] = dict(
            dest=dest,
            centroid_local=torch.tensor(pts[oh.vertices].mean(0), dtype=torch.float32, device=dev),
            planes=dest_cache[dest],
            task=task,
        )
    DIAG["meta"].update(objects=list(info), num_envs=env.num_envs)
    tm = env.termination_manager
    orig_compute = tm.compute
    state = {"step": 0}

    def compute(*a, **k):
        dones = orig_compute(*a, **k)
        with torch.no_grad():
            w = env.arena_world
            row = {"t": state["step"], "ep_len": env.episode_length_buf.cpu().tolist(), "done": dones.cpu().tolist()}
            terms = {}
            for i, name in enumerate(tm._term_names):
                v = tm._term_dones[:, i]
                if bool(v.any()):
                    terms[name] = v.cpu().tolist()
            row["terms"] = terms
            for obj, inf in info.items():
                T_o, T_d = w.get_pose_w(obj), w.get_pose_w(inf["dest"])
                c_w = quat_apply(T_o[:, 3:], inf["centroid_local"].expand(env.num_envs, 3)) + T_o[:, :3]
                c_d = quat_apply_inverse(T_d[:, 3:], c_w - T_d[:, :3])
                in_hull = (c_d @ inf["planes"][:, :3].T + inf["planes"][:, 3]).amax(-1) <= 0
                fm = scene[f"diag_finger_{obj}"].data.force_matrix_w.torch
                f_l, f_r = fm[:, 0, 0].norm(dim=-1), fm[:, 0, 1].norm(dim=-1)
                dest_force = scene[f"contact_sensor_{obj}"].data.force_matrix_w.torch[:, 0, 0, :]
                any_force = scene[f"diag_any_{obj}"].data.net_forces_w.torch[:, 0, :]
                over = object_bounds_center_over_destination(
                    object_centroid_W=w.get_centroid_w(obj),
                    T_W_D=T_d,
                    destination_bounds_D=w.get_aabb_in_local_frame(inf["dest"]),
                )
                mean_speed = w.get_mean_linear_velocity_w(obj).norm(dim=-1)
                row[obj] = {
                    "in_hull": in_hull.cpu().tolist(),
                    "f_left": f_l.cpu().tolist(),
                    "f_right": f_r.cpu().tolist(),
                    "dest_force": dest_force.cpu().tolist(),
                    "any_force": any_force.cpu().tolist(),
                    "over": over.cpu().tolist(),
                    "mean_speed": mean_speed.cpu().tolist(),
                    "z": (T_o[:, 2] - scene.env_origins[:, 2]).cpu().tolist(),
                    "c_dest": c_d.cpu().tolist(),
                }
            DIAG["rows"].append(row)
        state["step"] += 1
        if state["step"] % 1500 == 0:
            _dump()
        return dones

    tm.compute = compute


def _dump():
    out = os.environ.get("DIAG_OUT")
    if not out:
        out_dir = sys.argv[sys.argv.index("--experiment_output_directory") + 1]
        out = os.path.join(out_dir, "multi_diag.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": DIAG["meta"], "rows": DIAG["rows"]}, f)
    print(f"[multi_diag] wrote {out} rows={len(DIAG['rows'])}", flush=True)


def _patch_runner():
    from isaaclab_arena.evaluation import run_execution

    orig_rollout = run_execution.rollout_policy

    def rollout(env, policy, num_steps, num_episodes):
        _install_hook(env.unwrapped)
        try:
            return orig_rollout(env, policy, num_steps, num_episodes)
        finally:
            _dump()

    run_execution.rollout_policy = rollout


def main():
    from isaaclab_arena.utils.isaaclab_utils import simulation_app as sa

    orig_enter = sa.SimulationAppContext.__enter__

    def enter(self):
        ret = orig_enter(self)
        _patch_task()
        _patch_runner()
        return ret

    sa.SimulationAppContext.__enter__ = enter
    from isaaclab_arena.evaluation import experiment_runner

    experiment_runner.main()


if __name__ == "__main__":
    sys.exit(main())
