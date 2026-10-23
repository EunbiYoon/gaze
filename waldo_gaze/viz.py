"""Draw learned gaze trajectories on held-out images (qualitative figure).

  python -m waldo_gaze.viz --config configs/waldo.yaml --specs clip_full clip_conj --n_images 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from .agent import GazePolicy, rollout
from .data import crop_boxes, level_cells, load_all_cache, load_config, train_ids
from .env import GlimpseEnv
from .run import fold_periph

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specs", nargs="+", default=["clip_full", "clip_conj"])
    ap.add_argument("--images", nargs="*", default=None, help="image ids (default: first n of each fold)")
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=None, help="level-0 start cell (default: centre)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    data = load_all_cache(cfg)
    res = cfg["results_dir"]
    with open(os.path.join(res, "splits.json")) as f:
        splits = json.load(f)
    fold_of = {i: s["fold"] for s in splits for i in s["test"]}
    pos = {k: i for i, k in enumerate(data["ids"])}
    ids = args.images or [s["test"][0] for s in splits][: args.n_images]
    levels = data["levels"]
    G0 = levels[0]
    start = args.start if args.start is not None else (G0 // 2) * G0 + G0 // 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    fig, axes = plt.subplots(len(ids), len(args.specs), figsize=(3.2 * len(args.specs), 2.4 * len(ids)), squeeze=False)
    for row, iid in enumerate(ids):
        fold = fold_of[iid]
        tr = np.array([pos[k] for k in train_ids(splits[fold], data)])
        periph = fold_periph(data, tr, cfg["grid"]["periph_level"], cfg["grid"]["periph_pca"])
        img = Image.open(data["meta"]["images"][iid]["path"]).convert("RGB")
        W, H = img.size
        for col, spec in enumerate(args.specs):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_axis_off()
            ck = os.path.join(res, "ckpt", f"{spec}_f{fold}_s{args.seed}.pt")
            if not os.path.exists(ck):
                ax.set_title(f"{spec}: no checkpoint", fontsize=7)
                continue
            D = 2 * data["emb"][0].shape[-1]
            n_cells, cell_dim = periph.shape[2] * periph.shape[3], periph.shape[1] + 2
            pol = GazePolicy(D, n_cells, cell_dim, len(levels) + 3, cfg["ppo"]["hidden"],
                             sim_prior=cfg["ppo"].get("sim_prior", 0.0), sim_index=periph.shape[1] - 1).to(dev)
            pol.load_state_dict(torch.load(ck, map_location=dev))
            pol.eval()
            zeros = [np.zeros(e.shape[:2], np.float32) for e in data["emb"]]
            env = GlimpseEnv(data["emb"], periph, zeros[-1], zeros, data["succ"][-1], levels, cfg["env"],
                             data["text"]["full"])
            rollout(env, pol, np.array([pos[iid]]), np.array([start]), dev, greedy=True)
            trace = [t[0] for t in env.trace[: int(env.info["steps"][0]) + 1]]
            n = len(trace)
            for k, (l, r, c) in enumerate(trace):
                G = levels[l]
                cell = level_cells(W, H, G)[r * G + c]
                x0, y0, x1, y1 = crop_boxes(cell[None], W, H, cfg["grid"]["context"])[0]
                shade = 0.25 + 0.75 * k / max(n - 1, 1)
                last = k == n - 1
                ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.0 if last else 0.8,
                                               ec="#eb6834" if last else (0.16, 0.47, 0.84, shade)))
            for b in np.load(os.path.join(cfg["cache_dir"], f"{iid}.npz"))["boxes"]:
                ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.4, ec="#1baf7a"))
            ok = bool(env.info["success"][0])
            ax.set_title(f"{spec} | {'found' if ok else 'missed'} in {int(env.info['steps'][0])} steps", fontsize=7)
    fig.tight_layout()
    out = os.path.join(res, "analysis", "fig_trajectories.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
