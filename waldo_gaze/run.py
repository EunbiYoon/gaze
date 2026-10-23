"""Stage 2: train one PPO gaze agent per (reward spec, fold, seed) and evaluate on held-out images.

Preemption-safe: a run is one atomic JSON; finished runs are skipped on restart. Work is sharded
across processes (one per GPU) with --shard/--nshards.

  python -m waldo_gaze.run --config configs/waldo.yaml --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import os
import time
import zlib

import numpy as np
import torch

from .agent import GazePolicy, ppo_update, rollout
from .data import atomic_write_json, load_all_cache, load_config, make_splits, train_ids
from .env import GlimpseEnv
from .rewards import build_scores, static_metrics


def fold_periph(data: dict, train_idx: np.ndarray, level: int, k: int) -> np.ndarray:
    """Peripheral map = PCA (fit on train-fold images only) of one level's glimpse embeddings."""
    E = data["emb"][level].astype(np.float32)                    # [n, Np, D]
    n, Np, D = E.shape
    G = int(round(np.sqrt(Np)))
    X = E[train_idx].reshape(-1, D)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = (E - mu) @ Vt[:k].T                                      # [n, Np, k]
    P /= np.maximum(P[train_idx].std((0, 1)), 1e-6)
    # query-conditioned channel: similarity of every peripheral cell to THIS image's target text,
    # z-scored within the image. Box-free (text only), so it is legal input for every reward spec.
    sim = np.einsum("ind,id->in", E, data["text"]["full"])
    sim = (sim - sim.mean(1, keepdims=True)) / np.maximum(sim.std(1, keepdims=True), 1e-6)
    P = np.concatenate([P, sim[..., None]], -1)
    return np.ascontiguousarray(P.transpose(0, 2, 1).reshape(n, k + 1, G, G)).astype(np.float32)


def episode_metrics(env, score_L: np.ndarray, levels: list[int]) -> dict:
    """Ground-truth evaluation of a finished batch of episodes (one row per episode)."""
    info = env.info
    G = levels[-1]
    stopped = info["stop_cell"] >= 0
    fail_stop = stopped & ~info["success"]
    near = np.zeros(env.n, bool)
    hacked = np.zeros(env.n, bool)
    proxy_at_stop = np.full(env.n, np.nan)
    best_gt = np.full(env.n, np.nan)
    for e in range(env.n):
        i = env.img[e]
        tgt = np.flatnonzero(env.succ_L[i])
        best_gt[e] = score_L[i][tgt].max()
        if stopped[e]:
            proxy_at_stop[e] = score_L[i][info["stop_cell"][e]]
        if fail_stop[e]:
            r, c = divmod(info["stop_cell"][e], G)
            tr, tc = tgt // G, tgt % G
            near[e] = np.min(np.maximum(np.abs(tr - r), np.abs(tc - c))) <= 1
            hacked[e] = proxy_at_stop[e] >= best_gt[e]
    return {"success": info["success"].copy(), "timeout": info["timeout"].copy(),
            "near_miss": near & fail_stop, "distractor": fail_stop & ~near, "hacked": hacked,
            "steps": info["steps"].copy(), "found": info["first_success_step"] >= 0,
            "found_not_stopped": (info["first_success_step"] >= 0) & ~info["success"],
            "proxy_at_stop": proxy_at_stop, "best_gt": best_gt, "img": env.img.copy()}


def agent_fixation_maps(env, levels: list[int]) -> np.ndarray:
    """Per episode, where the agent looked: each fixation after the start spreads unit mass over the
    peripheral cells its glimpse covers. Returns [n_env, Gp*Gp] (unnormalised)."""
    Gp = env.Gp
    out = np.zeros((env.n, Gp * Gp), np.float32)
    steps = env.info["steps"]
    ar = np.arange(Gp)
    for t, fx in enumerate(env.trace[1:], start=1):          # trace[0] = start cell, like the human
        live = t <= steps                                    # first (central) fixation is dropped
        if not live.any():
            break
        l, r, c = fx[:, 0], fx[:, 1], fx[:, 2]
        G = np.asarray(levels)[l]
        lo_r, lo_c = r * Gp // G, c * Gp // G
        hi_r = np.maximum(lo_r, ((r + 1) * Gp - 1) // G)
        hi_c = np.maximum(lo_c, ((c + 1) * Gp - 1) // G)
        fr = (ar[None] >= lo_r[:, None]) & (ar[None] <= hi_r[:, None])
        fc = (ar[None] >= lo_c[:, None]) & (ar[None] <= hi_c[:, None])
        foot = (fr[:, :, None] & fc[:, None, :]).reshape(env.n, -1).astype(np.float32)
        foot /= foot.sum(1, keepdims=True)
        out[live] += foot[live]
    return out


def gaze_agreement(fmap: np.ndarray, img: np.ndarray, data: dict) -> dict:
    """Pearson CC between the agent's pooled fixation map and the human fixation map, per image."""
    res = {}
    if data["gaze"] is None:
        return res
    for i in np.unique(img):
        if not data["has_gaze"][i]:
            continue
        a = fmap[img == i].sum(0)
        h = data["gaze"][i]
        if a.std() < 1e-9 or h.std() < 1e-9:
            res[int(i)] = 0.0
        else:
            res[int(i)] = float(np.corrcoef(a, h)[0, 1])
    return res


def per_image(m: dict, ids: list[str], horizon: int, cc: dict | None = None) -> list[dict]:
    rows = []
    for i in np.unique(m["img"]):
        sel = m["img"] == i
        succ_steps = m["steps"][sel & m["success"]]
        curve = [(float(np.sum(succ_steps <= k)) / sel.sum()) for k in range(1, horizon + 1)]
        rows.append({
            "image_id": ids[i], "n_ep": int(sel.sum()),
            **{k: float(np.mean(m[k][sel])) for k in
               ("success", "timeout", "near_miss", "distractor", "hacked", "found", "found_not_stopped")},
            "steps": float(np.mean(m["steps"][sel])),
            "proxy_at_stop": float(np.nanmean(m["proxy_at_stop"][sel])) if np.isfinite(m["proxy_at_stop"][sel]).any() else None,
            "best_gt": float(m["best_gt"][sel][0]),
            "success_by_step": curve,
            "gaze_cc": (cc or {}).get(int(i)),
        })
    return rows


def train_eval(spec_name, spec, scores, data, split, periph, cfg, seed, device, ckpt_path=None):
    ids = data["ids"]
    pos = {k: i for i, k in enumerate(ids)}
    tr = np.array([pos[k] for k in train_ids(split, data)])
    te = np.array([pos[k] for k in split["test"]])
    tr_base = np.array([pos[k] for k in split["train"]])
    levels = data["levels"]
    ec, pc = cfg["env"], cfg["ppo"]

    if spec["type"] == "oracle":
        # wrong STOP must not be free, otherwise "stop immediately anywhere" beats searching
        succ = data["succ"][-1].astype(np.float32)
        terminal = ec["oracle_scale"] * succ + (1.0 - succ) * ec.get("oracle_wrong_stop", -1.0)
        potential = [ec["oracle_scale"] * s for s in scores]
    else:
        terminal = scores[-1]
        potential = scores
    potential = [ec.get("shaping_scale", 1.0) * p for p in potential]
    env = GlimpseEnv(data["emb"], periph, terminal, potential, data["succ"][-1], levels, ec, data["text"]["full"])

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = 2 * data["emb"][0].shape[-1]          # fovea = [embedding, embedding * target-text]
    n_cells, cell_dim = periph.shape[2] * periph.shape[3], periph.shape[1] + 2
    policy = GazePolicy(D, n_cells, cell_dim, len(levels) + 3, pc["hidden"],
                        sim_prior=pc.get("sim_prior", 0.0), sim_index=periph.shape[1] - 1).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=pc["lr"])
    G0sq = levels[0] ** 2

    gaze_t = gaze_ok = None
    if spec.get("gaze_coef", 0.0) > 0:
        if data["gaze"] is None:
            raise ValueError(f"{spec_name}: gaze_coef > 0 but the dataset has no human fixations")
        gaze_t = torch.from_numpy(data["gaze"]).to(device)
        ok = data["has_gaze"].copy()
        ok[te] = False                       # never use held-out images' human gaze in training
        gaze_ok = torch.from_numpy(ok).to(device)

    curve = []
    t0 = time.time()
    for it in range(pc["iters"]):
        img = rng.choice(tr, size=pc["n_envs"])
        start = rng.integers(0, G0sq, size=pc["n_envs"])
        steps = rollout(env, policy, img, start, device, greedy=False, rng=gen)
        stats = ppo_update(policy, opt, steps, pc, ec["gamma"], gen, device,
                           gaze=gaze_t, gaze_valid=gaze_ok, gaze_coef=float(spec.get("gaze_coef", 0.0)))
        if it % pc["log_every"] == 0 or it == pc["iters"] - 1:
            row = {"iter": it, "return": float(env.ep_return.mean()),
                   "train_success": float(env.info["success"].mean()),
                   "timeout": float(env.info["timeout"].mean()),
                   "steps": float(env.info["steps"].mean()), **stats, "sec": time.time() - t0}
            curve.append(row)
            print(f"  [{spec_name} s{seed}] it {it:4d} ret {row['return']:+.2f} "
                  f"succ {row['train_success']:.2f} to {row['timeout']:.2f} ent {stats['ent']:.2f}", flush=True)

    policy.eval()
    # held-out images: greedy from every level-0 start cell (deterministic given the policy)
    img = np.repeat(te, G0sq)
    start = np.tile(np.arange(G0sq), len(te))
    rollout(env, policy, img, start, device, greedy=True)
    cc = gaze_agreement(agent_fixation_maps(env, levels), env.img, data)
    test_rows = per_image(episode_metrics(env, scores[-1], levels), ids, ec["horizon"], cc)
    # training images: overfitting gap
    k = cfg["eval"]["train_starts"]
    img = np.repeat(tr_base, k)
    start = np.random.default_rng(seed + 7).integers(0, G0sq, size=len(img))
    rollout(env, policy, img, start, device, greedy=True)
    train_rows = per_image(episode_metrics(env, scores[-1], levels), ids, ec["horizon"])

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(policy.state_dict(), ckpt_path)
    return {"curve": curve, "test": test_rows, "train": train_rows, "train_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--specs", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = cfg["results_dir"]
    os.makedirs(os.path.join(res_dir, "runs"), exist_ok=True)

    data = load_all_cache(cfg)
    splits = make_splits(data["base_ids"], cfg["protocol"]["n_folds"], cfg["protocol"]["split_seed"])
    if args.shard == 0:
        atomic_write_json(splits, os.path.join(res_dir, "splits.json"))

    specs = args.specs or list(cfg["specs"].keys())
    folds = args.folds if args.folds is not None else list(range(cfg["protocol"]["n_folds"]))
    seeds = args.seeds if args.seeds is not None else cfg["protocol"]["seeds"]
    runs = [(s, f, sd) for s in specs for f in folds for sd in seeds]
    mine = runs[args.shard::args.nshards]
    print(f"[run shard {args.shard}/{args.nshards}] {len(mine)} of {len(runs)} runs on {device}; "
          f"{len(data['base_ids'])} images (+{len(data['ids']) - len(data['base_ids'])} augmented, train only)", flush=True)

    pos = {k: i for i, k in enumerate(data["ids"])}
    periph_cache, score_cache = {}, {}
    for spec_name, fold, seed in mine:
        out = os.path.join(res_dir, "runs", f"{spec_name}_f{fold}_s{seed}.json")
        if os.path.exists(out):
            continue
        spec = cfg["specs"][spec_name]
        split = splits[fold]
        tr = np.array([pos[k] for k in train_ids(split, data)])
        te = np.array([pos[k] for k in split["test"]])
        tr_base = np.array([pos[k] for k in split["train"]])
        if fold not in periph_cache:
            periph_cache = {fold: fold_periph(data, tr, cfg["grid"]["periph_level"], cfg["grid"]["periph_pca"])}
        if (spec_name, fold) not in score_cache:
            score_cache = {(spec_name, fold): build_scores(spec_name, spec, data, tr, cfg, seed=fold)}
            static_path = os.path.join(res_dir, "static", f"{spec_name}_f{fold}.json")
            if not os.path.exists(static_path):
                sc, info = score_cache[(spec_name, fold)]
                atomic_write_json({"info": info, "test": static_metrics(sc, data, te),
                                   "train": static_metrics(sc, data, tr_base)}, static_path)
        scores, info = score_cache[(spec_name, fold)]
        run_seed = seed * 1000 + fold * 10 + (zlib.crc32(spec_name.encode()) % 7)
        ckpt = (os.path.join(res_dir, "ckpt", f"{spec_name}_f{fold}_s{seed}.pt")
                if seed in cfg["protocol"].get("save_ckpt_seeds", []) else None)
        print(f"[run] {spec_name} fold {fold} seed {seed} info {info}", flush=True)
        res = train_eval(spec_name, spec, scores, data, split, periph_cache[fold], cfg, run_seed, device, ckpt)
        res.update({"spec": spec_name, "fold": fold, "seed": seed, "info": info,
                    "config": {k: cfg[k] for k in ("env", "ppo", "grid", "encoder")}})
        atomic_write_json(res, out)
        print(f"[run] done {spec_name} f{fold} s{seed}: test success "
              f"{np.mean([r['success'] for r in res['test']]):.3f} ({res['train_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
