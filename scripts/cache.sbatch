#!/usr/bin/env bash
#SBATCH --job-name=cache
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/cache-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
echo "python: $(which python)"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
D=${DATASET:-coco_search18}
mkdir -p logs/$D
export OMP_NUM_THREADS=4
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g python -m waldo_gaze.cache --config configs/$D.yaml \
    --shard $g --nshards 2 --device cuda > logs/$D/cache_$g.log 2>&1 &
done
wait
if grep -q Traceback logs/$D/cache_*.log; then
  echo "[cache] worker FAILED:"; grep -h -A3 "Error" logs/$D/cache_*.log | tail -n 20; exit 1
fi
python -m waldo_gaze.cache --config configs/$D.yaml --meta_only
