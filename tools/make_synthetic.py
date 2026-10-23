"""Generate a synthetic COCO-style dataset for smoke-testing (NOT for the paper's results).

  python tools/make_synthetic.py --out data/synthetic --n 25 --W 960 --H 640 --people 120
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from waldo_gaze.synthetic import make_scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--W", type=int, default=960)
ap.add_argument("--H", type=int, default=640)
ap.add_argument("--people", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
rng = np.random.default_rng(a.seed)
images, anns = [], []
for k in range(a.n):
    img, (x0, y0, x1, y1) = make_scene(a.W, a.H, a.people, rng)
    fn = f"scene_{k:03d}.png"
    img.save(os.path.join(a.out, "images", fn))
    images.append({"id": k, "file_name": fn, "width": a.W, "height": a.H})
    anns.append({"id": k, "image_id": k, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
with open(os.path.join(a.out, "annotations.json"), "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "waldo"}]}, f)
print(f"wrote {a.n} scenes to {a.out}")
