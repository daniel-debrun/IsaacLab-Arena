# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarise a multi_diag.json: per-episode termination reasons and RoboLab-style vs Arena-style placement.

Usage: python robolab_probe/analyze_multi.py multi_diag.json [--seq obj1,obj2] [--contact]
(--seq: RoboLab in-order success; --contact: RoboLab require_contact_with=destination)
"""

import json
import math
import numpy as np
import sys

FORCE_TH, SPEED_TH, CONE = 0.1, 0.003, math.pi / 4


def main():
    path = sys.argv[1]
    seq = None
    need_contact = "--contact" in sys.argv  # RoboLab require_contact_with=container (|force from destination| > 0.1 N)
    if "--seq" in sys.argv:
        seq = sys.argv[sys.argv.index("--seq") + 1].split(",")
    d = json.load(open(path))
    objs, n_env, rows = d["meta"]["objects"], d["meta"]["num_envs"], d["rows"]
    seq = seq or objs
    episodes = []
    for e in range(n_env):
        cur = []
        for r in rows:
            cur.append(r)
            if r["done"][e]:
                episodes.append((e, cur))
                cur = []
        if len(cur) > 100:
            episodes.append((e, cur, "partial"))
    summary = {"n": 0, "arena_success": 0, "rl_success": 0}
    for item in episodes:
        e, ep = item[0], item[1]
        partial = len(item) > 2
        T = len(ep)
        ends = [k for k, v in ep[-1]["terms"].items() if v[e]]

        def arr(o, key):
            return np.array([r[o][key][e] for r in ep])

        info = {}
        for o in objs:
            hull = arr(o, "in_hull")
            det = (arr(o, "f_left") <= 0.1) & (arr(o, "f_right") <= 0.1)
            det_left = arr(o, "f_left") <= 0.1
            df = np.array([r[o]["dest_force"][e] for r in ep])
            fn = np.linalg.norm(df, axis=1)
            ang = np.arccos(np.clip(df[:, 2] / np.maximum(fn, 1e-9), -1, 1))
            up = (fn >= FORCE_TH) & (ang <= CONE)
            slow = arr(o, "mean_speed") < SPEED_TH
            over = arr(o, "over")
            arena = up & slow & over
            anyf = np.linalg.norm(np.array([r[o]["any_force"][e] for r in ep]), axis=1)
            info[o] = dict(
                hull=hull,
                placed=hull & det_left & ((fn > FORCE_TH) if need_contact else True),
                arena=arena,
                up=up,
                slow=slow,
                over=over,
                anyf=anyf,
                z=arr(o, "z"),
            )
        # RoboLab-style: all objects currently placed (hull & left finger detached) at the same step
        allp = np.all([info[o]["placed"] for o in objs], axis=0)
        # in-order latch variant
        first = {o: np.argmax(info[o]["placed"]) if info[o]["placed"].any() else -1 for o in objs}
        ordered = all(first[a] >= 0 and first[b] >= 0 and first[a] <= first[b] for a, b in zip(seq[:-1], seq[1:]))
        rl_succ_step = int(np.argmax(allp)) if (allp.any() and ordered) else -1
        ok = "arena_succ" if "success" in ends else "-"
        print(f"env{e} T={T:5d}{' PARTIAL' if partial else ''} end={ends} {ok} RL-style success step={rl_succ_step}")
        for o in objs:
            i = info[o]
            fh = int(np.argmax(i["hull"])) if i["hull"].any() else -1
            fa = int(np.argmax(i["arena"])) if i["arena"].any() else -1
            nh = int(i["hull"].sum())
            print(
                f"    {o:24s} first_hull={fh:5d} first_arenapred={fa:5d} hull_steps={nh:5d} "
                f"hull&!up={int((i['hull'] & ~i['up']).sum()):5d} hull&!slow={int((i['hull'] & ~i['slow']).sum()):5d} "
                f"hull&!over={int((i['hull'] & ~i['over']).sum()):5d} hull&!up&anyforce={int((i['hull'] & ~i['up'] & (i['anyf'] > 0.1)).sum()):5d} "
                f"final_hull={bool(i['hull'][-1])} min_z={i['z'].min():.3f}"
            )
        if not partial:
            summary["n"] += 1
            summary["arena_success"] += "success" in ends
            summary["rl_success"] += rl_succ_step >= 0
        else:
            summary["partial_rl_success"] = summary.get("partial_rl_success", 0) + (rl_succ_step >= 0)
    print(summary)


if __name__ == "__main__":
    main()
