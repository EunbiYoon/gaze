"""Reward specifications.

Every spec maps each glimpse (image, level, cell) to a scalar score. The agent is trained on the
spec's finest-level score; it is always EVALUATED on ground truth. Scores are z-normalised within
(image, level) using only that image's own glimpses (label-free, so legal on test images).

Pessimism (kappa > 0) follows the uncertainty-penalised reward idea: score = mean - kappa * std
over an ensemble (prompt paraphrases, or a bootstrap-over-images ensemble of learned reward heads).
kappa='auto' is chosen on TRAIN-fold images only (inner split for learned heads).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def zscore(x: np.ndarray, axis: int = 1) -> np.ndarray:
    mu = x.mean(axis, keepdims=True)
    sd = x.std(axis, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def gt_percentile(score_L: np.ndarray, succ_L: np.ndarray) -> np.ndarray:
    """Per image: fraction of non-target cells scored strictly below the best target cell (1 = perfect)."""
    out = np.empty(len(score_L))
    for i in range(len(score_L)):
        best = score_L[i][succ_L[i]].max()
        out[i] = np.mean(score_L[i][~succ_L[i]] < best)
    return out


def _prompt_z(data: dict, prompts: list[str]) -> list[np.ndarray]:
    """Per level [n_img, N_l, P]: z-scored cosine similarity of every glimpse to THIS image's target
    text under each prompt template (targets differ per image in RefCOCOg / COCO-Search18)."""
    missing = [p for p in prompts if p not in data["text"]]
    if missing:
        raise KeyError(f"prompt(s) {missing} not in cache; re-run the cache step after editing cfg.prompts")
    T = np.stack([data["text"][p] for p in prompts], -1)                    # [n_img, D, P]
    return [zscore(np.einsum("ind,idp->inp", e.astype(np.float32), T), axis=1) for e in data["emb"]]


def _attr_agg(data: dict, agg: str) -> list[np.ndarray]:
    """Per-attribute z-scores aggregated over each image's valid attributes (min = conjunction)."""
    A, M = data["attr"], data["attr_mask"]                                 # [n, A, D], [n, A]
    if not M.any(1).all():
        raise ValueError("clip_attr spec needs >= 1 attribute per image (dataset has none for some images)")
    out = []
    for e in data["emb"]:
        z = zscore(np.einsum("ind,iad->ina", e.astype(np.float32), A), axis=1)
        m = M[:, None, :]
        if agg == "min":
            out.append(np.where(m, z, np.inf).min(-1).astype(np.float32))
        else:
            out.append(((z * m).sum(-1) / m.sum(-1)).astype(np.float32))
    return out


def _head_features(e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Query-conditioned features for learned reward heads: elementwise product with the target text
    embedding, so one head works when every image has a different target (a learned CLIP metric)."""
    return e.astype(np.float32) * t[:, None, :]


def _pick_kappa(member_z_L: np.ndarray, succ_L: np.ndarray, grid: list[float]) -> float:
    """member_z_L: [n_img, N_L, M]. Choose kappa maximising mean target percentile (ties -> smaller)."""
    mu, sd = member_z_L.mean(-1), member_z_L.std(-1)
    best_k, best_v = grid[0], -np.inf
    for k in sorted(grid):
        v = gt_percentile(mu - k * sd, succ_L).mean()
        if v > best_v + 1e-9:
            best_k, best_v = k, v
    return float(best_k)


def fit_heads(data: dict, idx: np.ndarray, n_members: int, C: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap-over-images ensemble of logistic heads on finest-level embeddings."""
    E = _head_features(data["emb"][-1], data["text"]["full"])
    S = data["succ"][-1]
    Ws, bs = [], []
    groups = data["group"][idx]
    ug = np.unique(groups)
    while len(Ws) < n_members:
        bg = rng.choice(ug, size=len(ug), replace=True)          # resample IMAGES (with their flips)
        boot = np.concatenate([idx[groups == g] for g in bg])
        X = E[boot].reshape(-1, E.shape[-1]).astype(np.float32)
        y = S[boot].reshape(-1)
        if y.all() or not y.any():
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(X, y)
        Ws.append(clf.coef_[0].astype(np.float32))
        bs.append(np.float32(clf.intercept_[0]))
    return np.stack(Ws, 1), np.array(bs)                                     # [D, M], [M]


def _head_z(data: dict, W: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    t = data["text"]["full"]
    return [zscore(np.einsum("ind,dm->inm", _head_features(e, t), W) + b, axis=1) for e in data["emb"]]


def build_scores(name: str, spec: dict, data: dict, train_idx: np.ndarray, cfg: dict, seed: int):
    """Returns (scores: list per level [n_img, N_l] float32, info dict)."""
    t = spec["type"]
    info = {"spec": name, "type": t}
    if t == "oracle":
        return [s.astype(np.float32) for s in data["succ"]], info

    if t == "clip":
        z = _prompt_z(data, spec["prompts"])
        return [zl[..., 0].astype(np.float32) for zl in z], info

    if t == "clip_attr":
        return _attr_agg(data, spec["agg"]), info

    grid = cfg["protocol"]["kappa_grid"]
    if t == "clip_ens":
        z = _prompt_z(data, spec["prompts"])
        kappa = spec["kappa"]
        if kappa == "auto":
            kappa = _pick_kappa(z[-1][train_idx], data["succ"][-1][train_idx], grid)
        info["kappa"] = float(kappa)
        return [(zl.mean(-1) - kappa * zl.std(-1)).astype(np.float32) for zl in z], info

    if t == "learned":
        lc = cfg["learned"]
        rng = np.random.default_rng(10_000 + seed)
        kappa = spec["kappa"]
        if kappa == "auto":
            # inner 2-fold over TRAIN images: fit on one half, choose kappa on the other half
            ug = rng.permutation(np.unique(data["group"][train_idx]))   # split by image, never by flip
            hg = [set(ug[: len(ug) // 2]), set(ug[len(ug) // 2:])]
            halves = [np.array([i for i in train_idx if data["group"][i] in h]) for h in hg]
            mz_parts, s_parts = [], []
            for a, b in ((0, 1), (1, 0)):
                W, bb = fit_heads(data, halves[a], lc["n_members"], lc["C"], rng)
                mz_parts.append(_head_z(data, W, bb)[-1][halves[b]])
                s_parts.append(data["succ"][-1][halves[b]])
            kappa = _pick_kappa(np.concatenate(mz_parts), np.concatenate(s_parts), grid)
        W, bb = fit_heads(data, train_idx, lc["n_members"], lc["C"], rng)
        z = _head_z(data, W, bb)
        info["kappa"] = float(kappa)
        return [(zl.mean(-1) - kappa * zl.std(-1)).astype(np.float32) for zl in z], info

    raise ValueError(f"unknown spec type {t}")


# --------------------------------------------------------------------------- static (no-RL) diagnostics
def static_metrics(scores: list[np.ndarray], data: dict, idx: np.ndarray) -> list[dict]:
    """Per image: does the reward itself already prefer a non-target glimpse?
    top1        : argmax finest cell is a target cell (exhaustive search, 4^L * G0^2 glimpses)
    gt_pct      : fraction of non-target cells ranked below the best target cell
    n_better    : # non-target cells scoring strictly above the best target cell (distractor count)
    c2f_success : greedy coarse-to-fine search under the reward (G0^2 + 4(L-1) glimpses)"""
    levels = data["levels"]
    sL, gL = scores[-1], data["succ"][-1]
    out = []
    for i in idx:
        best = sL[i][gL[i]].max()
        G0 = levels[0]
        k = int(np.argmax(scores[0][i]))
        r, c = divmod(k, G0)
        for l in range(1, len(levels)):
            G = levels[l]
            cand = [(2 * r + dr, 2 * c + dc) for dr in (0, 1) for dc in (0, 1)]
            vals = [scores[l][i][rr * G + cc] for rr, cc in cand]
            r, c = cand[int(np.argmax(vals))]
        out.append({
            "image_id": data["ids"][i],
            "top1": bool(gL[i][int(np.argmax(sL[i]))]),
            "gt_pct": float(np.mean(sL[i][~gL[i]] < best)),
            "n_better": int(np.sum(sL[i][~gL[i]] > best)),
            "c2f_success": bool(gL[i][r * levels[-1] + c]),
            "chance": float(gL[i].mean()),
        })
    return out
