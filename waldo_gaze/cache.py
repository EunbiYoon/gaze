"""Stage 1: encode every glimpse of every image once (frozen encoder) and store GT labels.

Per image the cache holds: glimpse embeddings per zoom level, target-success labels per cell, the
text embeddings of THIS image's target under every prompt template, its attribute embeddings, and
(if the dataset has them) human search-fixation maps per level.

After this, the RL environment is a pure lookup table, so PPO runs in minutes per run.
Preemption-safe: each image is written atomically and skipped if already present.

  python -m waldo_gaze.cache --config configs/waldo.yaml --shard 0 --nshards 2 --device cuda
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

from .data import (atomic_write_json, cell_labels, crop_boxes, fixation_map, level_cells, load_coco,
                   load_config)
from .encoders import build_encoder

AUG_SEP = "__"
Image.MAX_IMAGE_PIXELS = None


class TextCache:
    """Encodes each distinct string once (many images share a category or template)."""

    def __init__(self, enc):
        self.enc, self.memo = enc, {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self.memo]
        if new:
            for t, e in zip(new, self.enc.encode_texts(new)):
                self.memo[t] = e.astype(np.float32)
        return np.stack([self.memo[t] for t in texts])


def format_prompt(tmpl: str, rec) -> str:
    return tmpl.format(query=rec.query, category=rec.category_text or rec.query)


def encode_image(enc, img: Image.Image, boxes: np.ndarray, cfg: dict, fixations=None) -> dict:
    W, H = img.size
    out = {"W": W, "H": H, "boxes": boxes.astype(np.float32)}
    bs = cfg["encoder"]["batch_size"]
    for l, G in enumerate(cfg["grid"]["levels"]):
        cells = level_cells(W, H, G)
        crops = crop_boxes(cells, W, H, cfg["grid"]["context"])
        embs = []
        for s in range(0, len(crops), bs):
            batch = [img.crop(tuple(b)).resize((enc.size, enc.size), Image.BICUBIC) for b in crops[s:s + bs]]
            embs.append(enc.encode_images(batch))
        succ, overlap = cell_labels(cells, boxes)
        out[f"emb_l{l}"] = np.concatenate(embs).astype(np.float16)
        out[f"succ_l{l}"] = succ
        out[f"overlap_l{l}"] = overlap
        if fixations is not None and len(fixations):
            out[f"gaze_l{l}"] = fixation_map(fixations, W, H, G)
    return out


def text_fields(rec, cfg: dict, tc: TextCache) -> dict:
    names = list(cfg["prompts"].keys())
    T = tc([format_prompt(cfg["prompts"][n], rec) for n in names])
    out = {f"txt_{n}": T[i] for i, n in enumerate(names)}
    A = int(cfg.get("max_attrs", 4))
    attrs = list(rec.attrs)[:A]
    D = T.shape[1]
    emb = np.zeros((A, D), np.float32)
    mask = np.zeros(A, bool)
    if attrs:
        tmpl = cfg.get("attr_template", "{attr}")
        emb[: len(attrs)] = tc([tmpl.format(attr=a) for a in attrs])
        mask[: len(attrs)] = True
    out["attr_emb"], out["attr_mask"] = emb, mask
    return out


def write_meta(cfg: dict, recs) -> None:
    """meta.json = every image with a valid cache entry. run_all.sh calls this once after all shards."""
    cache_dir = cfg["cache_dir"]
    present = {}
    for r in recs:
        if not os.path.exists(os.path.join(cache_dir, f"{r.image_id}.npz")):
            continue
        info = {"path": r.path, "query": r.query, "meta": r.meta}
        present[r.image_id] = {**info, "aug_of": None}
        aug = f"{r.image_id}{AUG_SEP}hflip"
        if cfg.get("augment", {}).get("hflip", False) and os.path.exists(os.path.join(cache_dir, f"{aug}.npz")):
            present[aug] = {**info, "aug_of": r.image_id}
    atomic_write_json({"images": present, "levels": cfg["grid"]["levels"],
                       "encoder": cfg["encoder"], "context": cfg["grid"]["context"]},
                      os.path.join(cache_dir, "meta.json"))
    n_base = sum(v["aug_of"] is None for v in present.values())
    print(f"[cache] meta: {n_base}/{len(recs)} images cached (+{len(present) - n_base} augmented)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--meta_only", action="store_true", help="only (re)write meta.json from cached files")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cache_dir = cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    recs = load_coco(cfg["data"]["ann"], cfg["data"]["images"], cfg["data"].get("category"),
                     cfg.get("target_defaults"))
    if args.meta_only:
        write_meta(cfg, recs)
        return
    enc = build_encoder(cfg, args.device)
    tc = TextCache(enc)
    flip = cfg.get("augment", {}).get("hflip", False)
    L = len(cfg["grid"]["levels"])

    mine = recs[args.shard::args.nshards]
    t0 = time.time()
    for k, r in enumerate(mine):
        path = os.path.join(cache_dir, f"{r.image_id}.npz")
        if os.path.exists(path):
            continue
        img = Image.open(r.path).convert("RGB")
        boxes = r.boxes.copy()
        fix = None if r.fixations is None else r.fixations.copy()
        if img.size != (r.width, r.height):          # annotation/image size mismatch -> rescale
            sx, sy = img.size[0] / r.width, img.size[1] / r.height
            print(f"[warn] {r.image_id}: size {img.size} != ann ({r.width},{r.height}); rescaling boxes/fixations")
            boxes *= np.array([sx, sy, sx, sy])
            if fix is not None:
                fix *= np.array([sx, sy])
        txt = text_fields(r, cfg, tc)
        variants = [(path, img, boxes, fix)]
        if flip:
            W = img.size[0]
            fb = boxes.copy()
            fb[:, [0, 2]] = W - boxes[:, [2, 0]]
            ff = None if fix is None else np.stack([W - fix[:, 0], fix[:, 1]], 1)
            variants.append((os.path.join(cache_dir, f"{r.image_id}{AUG_SEP}hflip.npz"),
                             img.transpose(Image.FLIP_LEFT_RIGHT), fb, ff))
        # write the augmented copy FIRST so that "base file exists" implies "all variants exist"
        for vpath, vimg, vboxes, vfix in variants[::-1]:
            out = encode_image(enc, vimg, vboxes, cfg, vfix)
            if not out[f"succ_l{L - 1}"].any():
                print(f"[warn] {r.image_id}: no finest-level success cell (box outside image?) - skipped")
                break
            out.update(txt)
            tmp = vpath + f".tmp{os.getpid()}.npz"
            np.savez(tmp, **out)
            os.replace(tmp, vpath)
        print(f"[cache shard {args.shard}] {k + 1}/{len(mine)} {r.image_id} {img.size} "
              f"{time.time() - t0:.0f}s", flush=True)

    write_meta(cfg, recs)


if __name__ == "__main__":
    main()
