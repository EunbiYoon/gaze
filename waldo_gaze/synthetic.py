"""Synthetic 'find the striped man' scenes for smoke-testing the pipeline end to end.

Each person has four binary attributes (striped shirt, glasses, bobble hat, blue jeans). Exactly one
person per scene has all four (the target); distractors carry random subsets, so partial-attribute
confusers exist by construction. NOT a substitute for the real Waldo data - only for checking code.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

ATTRS = ("stripes", "glasses", "hat", "jeans")
SKIN = (238, 198, 160)


def draw_person(dr: ImageDraw.ImageDraw, x: float, y: float, h: float, attrs: dict, rng) -> tuple:
    """Draw a person whose bounding box top-left is (x, y) and height h. Returns xyxy box."""
    w = 0.42 * h
    head_r = 0.11 * h
    hx, hy = x + w / 2, y + 0.16 * h
    if attrs.get("hat"):
        dr.rectangle([hx - head_r, y, hx + head_r, y + 0.06 * h], fill=(200, 20, 20))
        dr.rectangle([hx - head_r, y + 0.03 * h, hx + head_r, y + 0.06 * h], fill=(255, 255, 255))
        dr.ellipse([hx - 0.04 * h, y - 0.03 * h, hx + 0.04 * h, y + 0.03 * h], fill=(200, 20, 20))
    dr.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=SKIN)
    if attrs.get("glasses"):
        gr = 0.035 * h
        for s in (-1, 1):
            cx = hx + s * 0.045 * h
            dr.ellipse([cx - gr, hy - gr, cx + gr, hy + gr], outline=(0, 0, 0), width=max(1, int(h / 40)))
    top, bot = y + 0.28 * h, y + 0.62 * h
    if attrs.get("stripes"):
        n = 6
        for k in range(n):
            y0 = top + (bot - top) * k / n
            dr.rectangle([x, y0, x + w, y0 + (bot - top) / n], fill=(210, 20, 20) if k % 2 == 0 else (250, 250, 250))
    else:
        col = tuple(int(c) for c in rng.integers(30, 230, 3))
        dr.rectangle([x, top, x + w, bot], fill=col)
    pants = (40, 70, 200) if attrs.get("jeans") else tuple(int(c) for c in rng.integers(30, 230, 3))
    dr.rectangle([x + 0.05 * w, bot, x + 0.95 * w, y + h], fill=pants)
    return (x, y, x + w, y + h)


def make_scene(W: int, H: int, n_people: int, rng, p_attr: float = 0.25, n_partial: int | None = None):
    """n_partial=None: distractor attributes are i.i.d. Bernoulli(p_attr) (never all four).
    n_partial=k   : exactly k distractors share 3 of the 4 target attributes; every other distractor
                    has at most 1 attribute. This is the controlled 'partial-match' dose."""
    img = Image.new("RGB", (W, H), (int(rng.integers(150, 220)), int(rng.integers(150, 220)), int(rng.integers(120, 200))))
    dr = ImageDraw.Draw(img)
    for _ in range(n_people * 3):                      # background clutter
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        s = rng.uniform(5, 25)
        dr.rectangle([x, y, x + s, y + s], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    target_idx = int(rng.integers(n_people))
    others = [k for k in range(n_people) if k != target_idx]
    partial = set(rng.choice(others, size=min(n_partial or 0, len(others)), replace=False).tolist())
    target_box = None
    for k in range(n_people):
        h = rng.uniform(0.045, 0.09) * H
        x, y = rng.uniform(0, W - 0.42 * h), rng.uniform(0, H - h)
        if k == target_idx:
            attrs = {a: True for a in ATTRS}
        elif n_partial is None:
            attrs = {a: bool(rng.random() < p_attr) for a in ATTRS}
            if all(attrs.values()):
                attrs["glasses"] = False
        elif k in partial:
            missing = ATTRS[int(rng.integers(len(ATTRS)))]
            attrs = {a: a != missing for a in ATTRS}
        else:
            keep = ATTRS[int(rng.integers(len(ATTRS)))] if rng.random() < 0.5 else None
            attrs = {a: a == keep for a in ATTRS}
        box = draw_person(dr, x, y, h, attrs, rng)
        if k == target_idx:
            target_box = box
    # redraw the target last so it is never fully occluded
    rng2 = np.random.default_rng(0)
    draw_person(dr, target_box[0], target_box[1], target_box[3] - target_box[1], {a: True for a in ATTRS}, rng2)
    return img, target_box


def prototype_from_text(text: str, size: int = 128) -> Image.Image:
    """Render a single person whose attributes are the keywords in `text` (fake text encoder)."""
    t = text.lower()
    attrs = {
        "stripes": any(k in t for k in ("stripe", "striped")),
        "glasses": "glasses" in t,
        "hat": any(k in t for k in ("hat", "bobble")),
        "jeans": any(k in t for k in ("jeans", "trousers", "pants")),
    }
    if any(k in t for k in ("waldo", "wally")) and not any(attrs.values()):
        attrs = {a: True for a in ATTRS}
    img = Image.new("RGB", (size, size), (185, 185, 160))
    draw_person(ImageDraw.Draw(img), size * 0.3, size * 0.1, size * 0.8, attrs, np.random.default_rng(1))
    return img
