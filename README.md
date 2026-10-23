# Gaze-as-Policy: Reward Hacking in Learned Visual Search

**Question.** An RL agent learns *where to look* (saccade, zoom, stop) from a proxy reward: a CLIP
image-text similarity, or a reward model learned from a few labelled images. Does it find the
target, or does it learn to fixate look-alikes the proxy scores higher? Do conjunctive,
uncertainty-penalised (pessimistic), or human-gaze-supervised agents hack less?

**Core pipeline (dataset-agnostic).**
1. Cut each large image into multi-scale glimpses and embed them once with frozen CLIP. The RL
   environment then becomes a lookup table.
2. Build several reward functions from the same embeddings, against **each image's own target text**.
3. Train a PPO gaze policy on each reward.
4. Score the policy against **ground truth only**. Static metrics (does the reward itself rank the
   target first?) separate *misspecification* from *optimisation* (`hacked` = the agent stopped on a
   non-target the reward prefers).

## Datasets (one config each, one 4 h allocation each)

| config | data | what it adds |
|---|---|---|
| `waldo` | CS682 Where's-Waldo scans | the motivating extreme case (tiny target, hundreds of look-alikes) |
| `coco_search18` | COCO-Search18 target-present images | **human search fixations**: agent-vs-human gaze agreement, plus a CGL-style human-gaze coverage loss (`*_cgl` specs) |
| `refcocog` | RefCOCOg (umd), images with ≥2 same-category objects | scale (hundreds of images); the target is one *described* instance among same-category distractors |
| `controlled` | generated scenes, `tools/make_controlled.py` | dose-response: exactly *k* distractors share 3 of 4 target attributes, k ∈ {0, 4, 16, 48} |

> CLEVR-Ref+ has no released download (it must be rendered with Blender), so `controlled` plays
> its role: a synthetic benchmark where the partial-match dose is set by construction. The scenes
> are simple drawn figures, so state in the paper that CLIP is out of its natural domain there.

### Data preparation

```bash
# Waldo: CS682 COCO json + images
#   data/waldo/annotations.json, data/waldo/images/

# COCO-Search18 (https://saliency.tuebingen.ai/datasets/COCO-Search18/): TP fixations + TP images
python tools/convert_coco_search18.py \
  --fixations coco_search18_fixations_TP_train.json coco_search18_fixations_TP_validation.json \
  --images_root data/coco_search18/images --out data/coco_search18 --per_task 40

# RefCOCOg (refer toolkit files + COCO train2014 images)
python tools/convert_refcocog.py --refs "data/refcocog/refs(umd).p" \
  --instances data/refcocog/instances.json --images_root data/coco/train2014 \
  --out data/refcocog --max_images 800
# then set data.images in configs/refcocog.yaml to the train2014 folder

# Controlled scenes
python tools/make_controlled.py --out data/controlled --levels 0 4 16 48 --per_level 40
```

Each converter prints what it skipped. If a converter says the input format is unexpected, the
download differs from the documented format: inspect the first entry and adapt the field names.

## Reward specs

`oracle` (upper bound) · `clip_name` (category only) · `clip_full` (the target's own query) ·
`clip_mean_attr` vs `clip_conj` (mean vs **min** of per-attribute scores, i.e. a bag of attributes vs a
conjunction; Waldo/controlled only) · `clip_pe_mean` vs `clip_pe_pess` (prompt-template ensemble,
mean vs mean − κ·std) · `learned` vs `learned_pess` (bootstrap ensemble of query-conditioned
logistic heads fit on the train fold) · `*_cgl` (same reward, plus a coverage loss pulling the saccade
distribution toward human fixations; COCO-Search18 only, never using test images' gaze).

## Protocol (fixes the CS682 report's evaluation)

- **5-fold image-level CV.** Every image is tested exactly once. Mirrored copies are training-only
  and never cross folds.
- **Hyperparameters fixed in `configs/base.yaml` before running.** κ is the only data-chosen value,
  and it is chosen on train-fold images only.
- **3 seeds.** The unit is the image (seed-averaged), with 95% bootstrap CIs, paired bootstrap +
  Wilcoxon for comparisons (the Bonferroni threshold is printed), and seed std.
- `analyze.py` prints a warning banner while the run grid is incomplete.

## Running on Unity

```bash
bash scripts/setup_env.sh                      # once, on the login node (env + CLIP weights)

salloc --job-name=m4 --partition=gpu-preempt --nodes=1 --gres=gpu:l40s:2 \
       --cpus-per-task=8 --mem=16G --time=04:00:00
source scripts/activate.sh
DATASET=controlled QUICK=1 bash scripts/run_all.sh   # ~15 min sanity pass first
DATASET=controlled bash scripts/run_all.sh           # full grid -> results/controlled/analysis/

# or batch with auto-requeue (finished work is skipped on restart):
mkdir -p logs && sbatch --export=ALL,DATASET=coco_search18 scripts/job.sbatch
```

Suggested order: `controlled` (fastest, sanity of the phenomenon), then `waldo`, `refcocog`, and
`coco_search18`. Budget figures are estimates, not GPU measurements; read `logs/<dataset>/run_*.log`
after the first runs. If one run takes more than ~2.5 min, lower `ppo.iters` or use
`EXTRA="protocol.seeds=[0,1]"` for a first pass.

## Outputs per dataset (`results/<dataset>/analysis/`)

- `summary.md`, `summary.json`, `table_main.tex`: main table, paired tests, single-prompt
  diagnostic, gaze agreement (COCO-Search18), and dose-response table (controlled).
- `fig_success_hacking.pdf`, `fig_budget.pdf`, `fig_dose_response.pdf` (controlled),
  `fig_trajectories.pdf`.

## Validation done so far

Every config was run end to end on CPU with a fake encoder, a numpy torch stand-in, and mock files
in each dataset's documented format. Checks: numerical gradient check of the PPO loss, a scripted
searcher reaching 100% success in the env, PPO learning with a location cue (0 → 44% train
success), and the gaze loss decreasing. **Not yet run:** real CLIP, GPU, or real data. Run the
QUICK pass first.
