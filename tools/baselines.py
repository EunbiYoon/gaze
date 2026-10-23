"""Training-free search baselines on a cached dataset (no RL, no boxes used except for scoring).
random(b): inspect b random finest cells; clip_greedy(b): visit peripheral cells in order of CLIP
text similarity, inspecting their finest cells until b are inspected. Both answer with the inspected
cell of highest CLIP similarity. answer = answer cell has the target; found = target was inspected."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from waldo_gaze.data import load_all_cache, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
ap.add_argument("--random_reps", type=int, default=50)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
data = load_all_cache(cfg)
levels = data["levels"]; pl = cfg["grid"]["periph_level"]
Gp, GL = levels[pl], levels[-1]; k = GL // Gp
base = [i for i, key in enumerate(data["ids"]) if data["aug_of"][key] is None]
T = data["text"]["full"]; rng = np.random.default_rng(0)
children = []
for pr in range(Gp):
    for pc in range(Gp):
        rows = np.arange(pr * k, (pr + 1) * k); cols = np.arange(pc * k, (pc + 1) * k)
        children.append((rows[:, None] * GL + cols[None, :]).ravel())
children = np.array(children)
N = GL * GL
budgets = sorted({min(b, N) for b in a.budgets} | {N})
res = {m: {b: {"answer": [], "found": []} for b in budgets} for m in ("random", "clip_greedy")}
chance = []
for i in base:
    sim_f = data["emb"][-1][i].astype(np.float32) @ T[i]
    sim_p = data["emb"][pl][i].astype(np.float32) @ T[i]
    succ = data["succ"][-1][i]; chance.append(succ.mean())
    order = children[np.argsort(-sim_p)].ravel()
    for b in budgets:
        seen = order[:b]; ans = seen[np.argmax(sim_f[seen])]
        res["clip_greedy"][b]["answer"].append(float(succ[ans]))
        res["clip_greedy"][b]["found"].append(float(succ[seen].any()))
        ra, rf = [], []
        for _ in range(a.random_reps):
            seen = rng.choice(N, size=b, replace=False)
            ra.append(succ[seen[np.argmax(sim_f[seen])]]); rf.append(succ[seen].any())
        res["random"][b]["answer"].append(float(np.mean(ra)))
        res["random"][b]["found"].append(float(np.mean(rf)))
def ci(x):
    x = np.asarray(x); bs = x[np.random.default_rng(1).integers(0, len(x), (5000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {"n_images": len(base), "grid": levels, "finest_cells": N, "chance": float(np.mean(chance)),
       "methods": {m: {str(b): {kk: ci(v) for kk, v in d.items()} for b, d in r.items()} for m, r in res.items()}}
od = os.path.join(cfg["results_dir"], "analysis"); os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "baselines.json"), "w"), indent=1)
print(f"{len(base)} images, finest grid {GL}x{GL} ({N} cells), chance per cell {out['chance'] * 100:.2f}%")
print(f"{'budget':>7} | {'random answer':>14} {'random found':>13} | {'CLIP-greedy answer':>19} {'CLIP-greedy found':>18}")
for b in budgets:
    r, g = out["methods"]["random"][str(b)], out["methods"]["clip_greedy"][str(b)]
    tag = "  <- exhaustive (CLIP argmax)" if b == N else ""
    print(f"{b:>7} | {r['answer'][0]*100:13.1f}% {r['found'][0]*100:12.1f}% | {g['answer'][0]*100:18.1f}% {g['found'][0]*100:17.1f}%{tag}")
