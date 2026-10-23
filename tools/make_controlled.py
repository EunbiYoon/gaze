"""Controlled partial-match benchmark (stand-in for CLEVR-Ref+, which must be rendered with Blender).

One target per scene carries 4 attributes; exactly `n_partial` distractors share 3 of them, all other
distractors carry at most 1. Scenes are generated for several n_partial levels so that hacking can be
plotted as a dose-response curve (analysis.group_by: n_partial).

  python tools/make_controlled.py --out data/controlled --levels 0 4 16 48 --per_level 40
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from waldo_gaze.synthetic import make_scene  # noqa: E402

ATTR_TEXT = ["a red and white striped shirt", "round black glasses", "a red and white bobble hat", "blue jeans"]
QUERY = "a cartoon man wearing " + ", ".join(ATTR_TEXT[:-1]) + " and " + ATTR_TEXT[-1]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/controlled")
ap.add_argument("--levels", type=int, nargs="+", default=[0, 4, 16, 48])
ap.add_argument("--per_level", type=int, default=40)
ap.add_argument("--W", type=int, default=1280)
ap.add_argument("--H", type=int, default=854)
ap.add_argument("--people", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
rng = np.random.default_rng(a.seed)
images, anns = [], []
for lvl in a.levels:
    for j in range(a.per_level):
        k = len(images)
        img, (x0, y0, x1, y1) = make_scene(a.W, a.H, a.people, rng, n_partial=lvl)
        fn = f"p{lvl:03d}_{j:03d}.png"
        img.save(os.path.join(a.out, "images", fn))
        images.append({"id": k, "file_name": fn, "width": a.W, "height": a.H, "query": QUERY,
                       "category_text": "a cartoon man", "attrs": ATTR_TEXT, "meta": {"n_partial": lvl}})
        anns.append({"id": k, "image_id": k, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
with open(os.path.join(a.out, "annotations.json"), "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "target"}]}, f)
print(f"wrote {len(images)} scenes ({a.per_level} per level, levels {a.levels}) to {a.out}")
