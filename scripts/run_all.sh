#!/usr/bin/env bash
# Run the whole pipeline inside an allocation (salloc or sbatch). Resumable: rerun after preemption.
#   DATASET=waldo bash scripts/run_all.sh            # full grid for configs/waldo.yaml
#   DATASET=coco_search18 QUICK=1 bash scripts/run_all.sh   # sanity pass: 2 folds, 1 seed, 40 PPO iters
# DATASET is one of: waldo coco_search18 refcocog controlled (one dataset per 4 h allocation)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=${DATASET:-waldo}
CFG=${CFG:-configs/$DATASET.yaml}
EXTRA=${EXTRA:-}
mkdir -p logs/$DATASET
NG=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
[ "$NG" -ge 1 ] || { echo "no GPU visible"; exit 1; }
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$(( NCPU / NG > 0 ? NCPU / NG : 1 ))
export TOKENIZERS_PARALLELISM=false
echo "[run_all] $NG GPU(s), $NCPU CPU(s), OMP_NUM_THREADS=$OMP_NUM_THREADS, cfg=$CFG"

RUN_ARGS=()
if [ "${QUICK:-0}" = "1" ]; then
  EXTRA="$EXTRA ppo.iters=40 protocol.seeds=[0] results_dir=results_quick/$DATASET"
  RUN_ARGS=(--folds 0 1)
fi
SET=(); [ -n "$EXTRA" ] && SET=(--set $EXTRA)

echo "[1/4] caching glimpse embeddings"
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -m waldo_gaze.cache --config "$CFG" --shard $g --nshards $NG --device cuda \
    ${SET[@]+"${SET[@]}"} > logs/$DATASET/cache_$g.log 2>&1 &
done
wait
python -m waldo_gaze.cache --config "$CFG" --meta_only ${SET[@]+"${SET[@]}"} | tee logs/$DATASET/cache_meta.log
grep -h -i "warn\|error\|Traceback" logs/$DATASET/cache_*.log || true
if grep -q Traceback logs/$DATASET/cache_*.log; then
  echo "[run_all] caching FAILED - stopping"; tail -n 3 logs/$DATASET/cache_*.log; exit 1
fi

echo "[2/4] training PPO gaze agents (one worker per GPU)"
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -m waldo_gaze.run --config "$CFG" --shard $g --nshards $NG \
    ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} ${SET[@]+"${SET[@]}"} > logs/$DATASET/run_$g.log 2>&1 &
done
wait

echo "[3/4] analysis"
python -m waldo_gaze.analyze --config "$CFG" ${SET[@]+"${SET[@]}"} | tee logs/$DATASET/analyze.log

echo "[4/4] trajectory figure"
python -m waldo_gaze.viz --config "$CFG" --specs clip_full learned learned_pess ${SET[@]+"${SET[@]}"} || true
echo "[run_all] done"
