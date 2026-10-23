"""Image/text encoders producing L2-normalised embeddings in a shared space.

CLIPEncoder : real experiments (HuggingFace transformers CLIP, frozen, fp16 on GPU).
FakeEncoder : hand-crafted features + keyword prototypes, CPU only, for smoke tests on synthetic data.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class CLIPEncoder:
    def __init__(self, name: str, device: str = "cuda", hf_cache: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(name, cache_dir=hf_cache).to(device, dtype=self.dtype).eval()
        self.tok = CLIPTokenizer.from_pretrained(name, cache_dir=hf_cache)
        self.size = int(self.model.config.vision_config.image_size)
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _as_tensor(out):
        # transformers versions differ: tensor, or a ModelOutput carrying the projected embeds
        if hasattr(out, "norm"):
            return out
        for key in ("image_embeds", "text_embeds", "pooler_output"):
            if getattr(out, key, None) is not None:
                return getattr(out, key)
        raise TypeError(f"Unexpected CLIP output type {type(out)}")

    def encode_images(self, crops: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        arr = np.stack([np.asarray(c, dtype=np.float32) / 255.0 for c in crops])     # crops already size x size
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        x = ((x - self.mean) / self.std).to(self.dtype)
        with torch.no_grad():
            f = self._as_tensor(self.model.get_image_features(pixel_values=x)).float()
        return _l2(f.cpu().numpy())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        t = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self._as_tensor(self.model.get_text_features(**t)).float()
        return _l2(f.cpu().numpy())


class FakeEncoder:
    """Colour / stripe / dark-ring / blue features. Text -> features of a rendered prototype."""

    size = 64

    @staticmethod
    def _feat(img: Image.Image) -> np.ndarray:
        a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        hist = np.histogramdd(a.reshape(-1, 3), bins=(3, 3, 3), range=((0, 256),) * 3)[0].ravel()
        hist = hist / hist.sum()
        redness = (r - (g + b) / 2) / 255.0
        prof = redness.mean(1) - redness.mean()
        spec = np.abs(np.fft.rfft(prof))[1:17]
        bands = np.array([spec[i:i + 4].sum() for i in range(0, 16, 4)]) / 8.0
        dark = np.mean(a.max(-1) < 45) * 10
        blue = np.mean((b > r + 50) & (b > g + 30)) * 5
        red = np.mean((r > 170) & (g < 80) & (b < 80)) * 5
        white = np.mean(a.min(-1) > 235) * 5
        return np.concatenate([hist, bands, [dark, blue, red, white]]).astype(np.float32)

    def encode_images(self, crops: list[Image.Image]) -> np.ndarray:
        return _l2(np.stack([self._feat(c) for c in crops]))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        from .synthetic import prototype_from_text
        return _l2(np.stack([self._feat(prototype_from_text(t)) for t in texts]))


def build_encoder(cfg: dict, device: str):
    e = cfg["encoder"]
    if e["type"] == "clip":
        return CLIPEncoder(e["name"], device=device, hf_cache=e.get("hf_cache"))
    if e["type"] == "fake":
        return FakeEncoder()
    raise ValueError(f"unknown encoder type {e['type']}")
