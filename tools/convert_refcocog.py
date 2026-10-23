"""RefCOCOg (umd) -> this repo's COCO-style json. One referred object per image, kept only if the
image contains at least `--min_same` objects of the referred object's category (so same-category
distractors exist and the description has to do the work).

Inputs from the refer toolkit release (github.com/lichengunc/refer): refcocog/refs(umd).p and
refcocog/instances.json; images are COCO train2014.

  python tools/convert_refcocog.py --refs "data/refcocog/refs(umd).p" \
      --instances data/refcocog/instances.json --images_root data/coco/train2014 \
      --out data/refcocog --max_images 800
"""
import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--refs", required=True)
ap.add_argument("--instances", required=True)
ap.add_argument("--images_root", required=True)
ap.add_argument("--out", default="data/refcocog")
ap.add_argument("--max_images", type=int, default=800)
ap.add_argument("--min_same", type=int, default=2, help="min # objects of the target's category in the image")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

with open(a.refs, "rb") as f:
    refs = pickle.load(f, encoding="latin1")          # the release pickles are Python-2 era
with open(a.instances) as f:
    inst = json.load(f)
ims = {im["id"]: im for im in inst["images"]}
anns = {an["id"]: an for an in inst["annotations"]}
cats = {c["id"]: c["name"] for c in inst["categories"]}
per_img_cat = defaultdict(Counter)
for an in inst["annotations"]:
    if not an.get("iscrowd", 0):
        per_img_cat[an["image_id"]][an["category_id"]] += 1

rng = np.random.default_rng(a.seed)
order = rng.permutation(len(refs))
seen, images, out_anns, skipped = set(), [], [], Counter()
for j in order:
    r = refs[j]
    iid, aid = r["image_id"], r["ann_id"]
    if iid in seen:
        skipped["dup_image"] += 1
        continue
    an = anns.get(aid)
    if an is None or an.get("iscrowd", 0):
        skipped["no_ann"] += 1
        continue
    if per_img_cat[iid][an["category_id"]] < a.min_same:
        skipped["no_same_cat_distractor"] += 1
        continue
    if not os.path.exists(os.path.join(a.images_root, ims[iid]["file_name"])):
        skipped["image_missing"] += 1
        continue
    sent = r["sentences"][int(rng.integers(len(r["sentences"])))]
    text = (sent.get("raw") or sent.get("sent") or "").strip().rstrip(".")
    cname = cats[an["category_id"]]
    article = "an" if cname[0].lower() in "aeiou" else "a"
    seen.add(iid)
    images.append({"id": f"rcg_{iid}", "file_name": ims[iid]["file_name"],
                   "width": ims[iid]["width"], "height": ims[iid]["height"],
                   "query": text, "category_text": f"{article} {cname}",
                   "meta": {"n_same_category": per_img_cat[iid][an["category_id"]], "ref_split": r.get("split"),
                            "category": cname}})
    out_anns.append({"id": len(out_anns), "image_id": f"rcg_{iid}", "category_id": 1,
                     "bbox": an["bbox"], "iscrowd": 0})
    if len(images) >= a.max_images:
        break

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "annotations.json"), "w") as f:
    json.dump({"images": images, "annotations": out_anns, "categories": [{"id": 1, "name": "target"}]}, f)
print(f"wrote {len(images)} images to {a.out}/annotations.json; skipped: {dict(skipped)}")
print(f"set data.images: {a.images_root} in configs/refcocog.yaml")
