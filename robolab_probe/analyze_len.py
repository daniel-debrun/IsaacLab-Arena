"""Summarise len_diag.py output: per episode, when do Arena's and RoboLab's in-container conditions hold for both hammers."""
import json, sys
d = json.load(open(sys.argv[1]))
rows, meta = d["rows"], d["meta"]
objs = [k for k in rows[0]["objs"]]
N = meta["num_envs"]
eps = [[] for _ in range(N)]  # list of episodes per env: list of row indices
cur = [[] for _ in range(N)]
for i, r in enumerate(rows):
    for e in range(N):
        cur[e].append(i)
        if r["done"][e]:
            eps[e].append(cur[e]); cur[e] = []
for e in range(N):
    if cur[e]:
        eps[e].append(cur[e])
VT = 0.003
print("objs", objs)
for e in range(N):
    for k, idxs in enumerate(eps[e]):
        end = rows[idxs[-1]]
        arena_both = rl_both = rl_nogrip_both = 0
        first_arena = first_rl = None
        stats = {o: dict(arena=0, hull=0, hull_det=0, over=0, up=0, slow=0) for o in objs}
        for j, i in enumerate(idxs):
            o = rows[i]["objs"]
            ar = {n: o[n]["over"][e] and o[n]["up"][e] and o[n]["speed"][e] < VT for n in objs}
            rl = {n: o[n]["in_hull"][e] and o[n]["f_grip"][e] < 0.1 for n in objs}
            for n in objs:
                s = stats[n]
                s["arena"] += ar[n]; s["hull"] += o[n]["in_hull"][e]; s["hull_det"] += rl[n]
                s["over"] += o[n]["over"][e]; s["up"] += o[n]["up"][e]; s["slow"] += o[n]["speed"][e] < VT
            if all(ar.values()):
                arena_both += 1; first_arena = first_arena if first_arena is not None else j
            if all(rl.values()):
                rl_both += 1; first_rl = first_rl if first_rl is not None else j
        fin = {n: (end["objs"][n]["in_hull"][e], end["objs"][n]["over"][e], end["objs"][n]["up"][e],
                   round(end["objs"][n]["speed"][e], 4), round(end["objs"][n]["z"][e], 3)) for n in objs}
        print(f"env{e} ep{k} len={len(idxs)} arena_both_steps={arena_both}(first {first_arena}) robolab_both_steps={rl_both}(first {first_rl}) "
              f"final(hull,over,up,speed,z)={fin}")
        for n in objs:
            print("    ", n, stats[n])
