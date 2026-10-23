#!/usr/bin/env bash
# One-time setup on the LOGIN node (has internet): env + CLIP weights into ./hf_cache.
set -euo pipefail
cd "$(dirname "$0")/.."
module load conda/latest 2>/dev/null || true
conda create -y -n waldo python=3.11
conda activate waldo
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export HF_HOME=$PWD/hf_cache
python - <<'PY'
from transformers import CLIPModel, CLIPTokenizer
CLIPModel.from_pretrained("openai/clip-vit-base-patch32"); CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP weights cached in", __import__("os").environ["HF_HOME"])
PY
