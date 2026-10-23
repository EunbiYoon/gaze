"""Data loading, glimpse-grid geometry, ground-truth labels, and image-level CV splits.

Dataset format = COCO json with optional per-image target fields on each `images` entry:
  "query":         text describing THIS image's target   (e.g. RefCOCOg sentence, "a clock")
  "category_text": coarse name of the target class       (e.g. "a person", "a clock")
  "attrs":         list of attribute phrases              (used by conjunctive rewards)
  "fixations":     [[x, y], ...] human search fixations in image pixels (COCO-Search18)
  "meta":          free-form dict carried into the results (e.g. {"n_partial": 8})
Missing fields fall back to cfg["target_defaults"] (used for Waldo, where every image has the same target).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml


# --------------------------------------------------------------------------- config
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict) and k != "specs":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)          # 'specs' is replaced wholesale per dataset
    return out


def _read_yaml(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = _deep_merge(_read_yaml(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = _read_yaml(path)
    for ov in overrides or []:            # e.g. ppo.iters=30
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    cfg = _coerce_numbers(cfg)        # YAML 1.1 reads "1e-3" as a string
    levels = cfg["grid"]["levels"]
    for a, b in zip(levels, levels[1:]):
        if b != 2 * a:
            raise ValueError(f"grid.levels must double each level, got {levels}")
    return cfg


def _coerce_numbers(x):
    if isinstance(x, dict):
        return {k: _coerce_numbers(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce_numbers(v) for v in x]
    if isinstance(x, str) and x.strip():
        try:
            return float(x) if any(ch in x.lower() for ch in ".e") and x.strip()[0] in "+-.0123456789" else x
        except ValueError:
            return x
    return x


def atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- COCO
@dataclass
class ImageRec:
    image_id: str
    path: str
    width: int
    height: int
    boxes: np.ndarray                     # [K, 4] xyxy in pixels
    query: str = ""
    category_text: str = ""
    attrs: list = field(default_factory=list)
    fixations: np.ndarray | None = None   # [F, 2] xy pixels, or None
    meta: dict = field(default_factory=dict)


def load_coco(ann_path: str, images_dir: str, category: str | None = None,
              defaults: dict | None = None) -> list[ImageRec]:
    defaults = defaults or {}
    with open(ann_path) as f:
        d = json.load(f)
    cat_names = {c["id"]: str(c.get("name", "")).lower() for c in d.get("categories", [])}
    boxes: dict = {}
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if category is not None and cat_names.get(a.get("category_id"), "") != category.lower():
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    recs = []
    for im in d["images"]:
        if im["id"] not in boxes:
            continue
        safe_id = str(im["id"]).replace("/", "_").replace(" ", "_")
        fx = im.get("fixations")
        recs.append(ImageRec(
            safe_id, os.path.join(images_dir, im["file_name"]), int(im["width"]), int(im["height"]),
            np.asarray(boxes[im["id"]], dtype=np.float64),
            query=im.get("query", defaults.get("query", "")),
            category_text=im.get("category_text", defaults.get("category", "")),
            attrs=list(im.get("attrs", defaults.get("attrs", []))),
            fixations=np.asarray(fx, dtype=np.float64).reshape(-1, 2) if fx else None,
            meta=dict(im.get("meta", {}))))
    recs.sort(key=lambda r: r.image_id)
    if not recs:
        raise RuntimeError(f"No annotated images found in {ann_path}")
    if any(not r.query for r in recs):
        raise RuntimeError("some images have no 'query' and cfg.target_defaults.query is empty")
    return recs


# --------------------------------------------------------------------------- geometry
def level_cells(W: float, H: float, G: int) -> np.ndarray:
    """Cell cores for a G x G grid, row-major (index = row * G + col). Returns [G*G, 4] xyxy."""
    xs = np.linspace(0, W, G + 1)
    ys = np.linspace(0, H, G + 1)
    x0 = np.tile(xs[:-1], G)
    x1 = np.tile(xs[1:], G)
    y0 = np.repeat(ys[:-1], G)
    y1 = np.repeat(ys[1:], G)
    return np.stack([x0, y0, x1, y1], 1)


def crop_boxes(cells: np.ndarray, W: float, H: float, context: float) -> np.ndarray:
    """Square crops centred on each cell (no aspect distortion), shifted inside the image."""
    cx = (cells[:, 0] + cells[:, 2]) / 2
    cy = (cells[:, 1] + cells[:, 3]) / 2
    side = np.maximum(cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 1]) * (1 + 2 * context)
    side = np.minimum(side, max(W, H))
    x0 = np.clip(cx - side / 2, 0, np.maximum(W - side, 0))
    y0 = np.clip(cy - side / 2, 0, np.maximum(H - side, 0))
    return np.stack([x0, y0, x0 + side, y0 + side], 1)


def cell_labels(cells: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """success: a GT centre lies in the cell core OR the cell centre lies inside a GT box
    (two-way pointing criterion, robust to both tiny and huge targets).
    overlap: max over GT boxes of |cell ∩ box| / |box| (fraction of the target visible in the core)."""
    gcx = (boxes[:, 0] + boxes[:, 2]) / 2
    gcy = (boxes[:, 1] + boxes[:, 3]) / 2
    ccx = (cells[:, 0] + cells[:, 2]) / 2
    ccy = (cells[:, 1] + cells[:, 3]) / 2
    c = cells[:, None, :]
    b = boxes[None, :, :]
    centre_in_cell = (gcx[None] >= c[..., 0]) & (gcx[None] < c[..., 2]) & \
                     (gcy[None] >= c[..., 1]) & (gcy[None] < c[..., 3])
    cell_in_box = (ccx[:, None] >= b[..., 0]) & (ccx[:, None] <= b[..., 2]) & \
                  (ccy[:, None] >= b[..., 1]) & (ccy[:, None] <= b[..., 3])
    success = (centre_in_cell | cell_in_box).any(1)
    iw = np.clip(np.minimum(c[..., 2], b[..., 2]) - np.maximum(c[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(c[..., 3], b[..., 3]) - np.maximum(c[..., 1], b[..., 1]), 0, None)
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    overlap = (iw * ih / np.maximum(area_b, 1e-9)).max(1)
    return success, overlap.astype(np.float32)


def fixation_map(fix: np.ndarray, W: float, H: float, G: int) -> np.ndarray:
    """Normalised histogram of fixations over a G x G grid (row-major)."""
    col = np.clip((fix[:, 0] / W * G).astype(int), 0, G - 1)
    row = np.clip((fix[:, 1] / H * G).astype(int), 0, G - 1)
    h = np.bincount(row * G + col, minlength=G * G).astype(np.float32)
    return h / max(h.sum(), 1.0)


# --------------------------------------------------------------------------- splits
def make_splits(image_ids: list[str], n_folds: int, seed: int) -> list[dict]:
    ids = sorted(image_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    folds = [sorted(ids[i] for i in perm[k::n_folds]) for k in range(n_folds)]
    out = []
    for k in range(n_folds):
        test = set(folds[k])
        out.append({"fold": k, "test": folds[k], "train": [i for i in ids if i not in test]})
    return out


def train_ids(split: dict, data: dict) -> list[str]:
    """Train-fold base images plus their augmented copies (augmented copies are never tested)."""
    keep = set(split["train"])
    return sorted(i for i in data["ids"] if (data["aug_of"][i] or i) in keep)


# --------------------------------------------------------------------------- cache loading
def load_all_cache(cfg: dict) -> dict:
    """Stack every cached image into dense arrays (all images share the grid layout)."""
    cache_dir = cfg["cache_dir"]
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    ids = sorted(meta["images"].keys())
    if not ids:
        raise RuntimeError(f"cache in {cache_dir} is empty (0 images): run the caching step on a GPU node first")
    missing = [i for i in ids if not os.path.exists(os.path.join(cache_dir, f"{i}.npz"))]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing from cache (e.g. {missing[:3]}); rerun cache step")
    L = len(cfg["grid"]["levels"])
    pl = cfg["grid"]["periph_level"]
    emb = [[] for _ in range(L)]
    succ = [[] for _ in range(L)]
    overlap = [[] for _ in range(L)]
    text, attr, attr_mask, gaze = {}, [], [], []
    for i in ids:
        z = np.load(os.path.join(cache_dir, f"{i}.npz"))
        for l in range(L):
            emb[l].append(z[f"emb_l{l}"])
            succ[l].append(z[f"succ_l{l}"])
            overlap[l].append(z[f"overlap_l{l}"])
        for k in z.files:
            if k.startswith("txt_"):
                text.setdefault(k[4:], []).append(z[k])
        attr.append(z["attr_emb"])
        attr_mask.append(z["attr_mask"])
        gaze.append(z[f"gaze_l{pl}"] if f"gaze_l{pl}" in z.files else None)
    aug_of = {i: meta["images"][i].get("aug_of") for i in ids}
    base_pos = {i: k for k, i in enumerate(ids)}
    group = np.array([base_pos[aug_of[i] or i] for i in ids])     # image and its flips share a group
    has_gaze = np.array([g is not None for g in gaze])
    Np = cfg["grid"]["levels"][pl] ** 2
    gz = np.stack([g if g is not None else np.zeros(Np, np.float32) for g in gaze]).astype(np.float32)
    return {
        "aug_of": aug_of,
        "group": group,
        "base_ids": [i for i in ids if aug_of[i] is None],
        "ids": ids,
        "meta": meta,
        "emb": [np.stack(e).astype(np.float16) for e in emb],     # [n_img, N_l, D]
        "succ": [np.stack(s).astype(bool) for s in succ],          # [n_img, N_l]
        "overlap": [np.stack(o).astype(np.float32) for o in overlap],
        "text": {k: np.stack(v).astype(np.float32) for k, v in text.items()},   # name -> [n_img, D]
        "attr": np.stack(attr).astype(np.float32),                 # [n_img, A, D]
        "attr_mask": np.stack(attr_mask).astype(bool),             # [n_img, A]
        "gaze": gz if has_gaze.any() else None,                    # [n_img, Gp*Gp] human fixation maps
        "has_gaze": has_gaze,
        "levels": list(cfg["grid"]["levels"]),
    }
