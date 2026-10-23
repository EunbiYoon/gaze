"""COCO-Search18 -> this repo's COCO-style json (target-present images + human search fixations).

Download from https://saliency.tuebingen.ai/datasets/COCO-Search18/ :
  - the TP fixation files (train + validation; test fixations are withheld by the authors)
  - the TP images archive

Fixation entries are dicts with: name, subject, task, condition, bbox [x, y, w, h], X, Y, T, length,
correct, split. Coordinates (bbox and X/Y) are in the 1680x1050 display space. If your images are a
different size, everything is rescaled to the real image size.

  python tools/convert_coco_search18.py \
      --fixations coco_search18_fixations_TP_train.json coco_search18_fixations_TP_validation.json \
      --images_root data/coco_search18/images --out data/coco_search18 --per_task 40

We pool fixations over the 10 subjects from CORRECT trials only and drop each scanpath's first
fixation (the central start position), as is standard. We use our own image-level CV folds; the
official split field is recorded in meta but not used.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

DISPLAY = (1680, 1050)

ap = argparse.ArgumentParser()
ap.add_argument("--fixations", nargs="+", required=True)
ap.add_argument("--images_root", required=True, help="folder with <task>/<name> or flat <name> images")
ap.add_argument("--out", default="data/coco_search18")
ap.add_argument("--per_task", type=int, default=40, help="images sampled per target category (0 = all)")
ap.add_argument("--min_fix", type=int, default=3, help="drop images with fewer pooled fixations")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

entries = []
for p in a.fixations:
    with open(p) as f:
        entries += json.load(f)
need = {"name", "task", "condition", "bbox", "X", "Y"}
if not entries or not need <= set(entries[0]):
    raise SystemExit(f"unexpected fixation format; first entry keys: {sorted(entries[0]) if entries else None}")

groups = defaultdict(list)
for e in entries:
    if str(e["condition"]).lower().startswith("present"):
        groups[(e["task"], e["name"])].append(e)
print(f"{len(entries)} scanpaths -> {len(groups)} target-present (task, image) pairs")


def find_image(task, name):
    for cand in (os.path.join(a.images_root, task, name), os.path.join(a.images_root, name)):
        if os.path.exists(cand):
            return cand
    return None


rng = np.random.default_rng(a.seed)
by_task = defaultdict(list)
for key in sorted(groups):
    by_task[key[0]].append(key)
chosen = []
for task, keys in sorted(by_task.items()):
    keys = list(keys)
    rng.shuffle(keys)
    chosen += keys[: a.per_task] if a.per_task > 0 else keys

images, anns, missing = [], [], 0
for task, name in chosen:
    path = find_image(task, name)
    if path is None:
        missing += 1
        continue
    W, H = Image.open(path).size
    sx, sy = W / DISPLAY[0], H / DISPLAY[1]
    sps = groups[(task, name)]
    fx = []
    lengths = []
    for e in sps:
        if int(e.get("correct", 1)) != 1:
            continue
        X, Y = list(e["X"])[1:], list(e["Y"])[1:]
        fx += [[x * sx, y * sy] for x, y in zip(X, Y)]
        lengths.append(len(e["X"]))
    if len(fx) < a.min_fix:
        continue
    x, y, w, h = sps[0]["bbox"]
    k = len(images)
    article = "an" if task[0].lower() in "aeiou" else "a"
    images.append({
        "id": f"{task.replace(' ', '_')}_{os.path.splitext(name)[0]}",
        "file_name": os.path.relpath(path, a.images_root), "width": W, "height": H,
        "query": f"{article} {task}", "category_text": f"{article} {task}",
        "fixations": fx,
        "meta": {"task": task, "human_fixations_to_target": float(np.mean(lengths)) if lengths else None,
                 "n_subjects": len(lengths), "official_split": sps[0].get("split")}})
    anns.append({"id": k, "image_id": images[-1]["id"], "category_id": 1,
                 "bbox": [x * sx, y * sy, w * sx, h * sy], "iscrowd": 0})

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "annotations.json"), "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "target"}]}, f)
print(f"wrote {len(images)} images ({len(by_task)} tasks) to {a.out}/annotations.json; "
      f"{missing} images not found under {a.images_root}")
print(f"set data.images: {a.images_root} in configs/coco_search18.yaml")
