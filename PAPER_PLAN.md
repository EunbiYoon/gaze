# Workshop paper plan (fix these before seeing results)

**Working title.** *Where Should the Agent Look? Reward Hacking in Learned Visual Search*

## Hypotheses and decision rules (pre-registered: write them in the paper as stated here)

| | Claim | Test | Supported if |
|---|---|---|---|
| H1 | The CLIP proxy is misspecified for Waldo | static top-1 and `n_better` per prompt (no RL) | top-1 CI upper bound < 50% for `clip_full`, and median `n_better` ≥ 1 |
| H2 | An RL gaze policy optimises the proxy into look-alikes | `hacked` rate for `clip_full` | hacked CI lower bound > 0 **and** RL success < oracle success (paired CI excludes 0) |
| H3a | Conjunction beats bag-of-attributes | paired `clip_conj − clip_mean_attr` | paired CI excludes 0 and Wilcoxon p < Bonferroni threshold |
| H3b | Pessimism reduces hacking | paired `*_pess − *_mean` on success and on hacked | same rule |
| H4 | The learned reward hacks more off-distribution | `learned` train-image success − held-out success | gap CI excludes 0 |
| H5 | Hacking grows with the number of partial-attribute look-alikes | `controlled`: hacked rate across n_partial levels for `clip_full` | monotone trend; CI at the largest level excludes the CI at 0 |
| H6 | Human-gaze supervision reduces hacking under the same proxy | `coco_search18`: paired `clip_full_cgl − clip_full` on success and hacked | same paired rule as H3 |
| H7 | Hacking agents look less like humans | `coco_search18`: gaze CC of hacked vs non-hacked runs/rewards, vs the centre-bias baseline | CC difference CI excludes 0 |
| G | The effects are not Waldo-specific | the H2/H3 direction holds on ≥2 of the 4 datasets | report per dataset; no pooling across datasets |

If a hypothesis fails, report it as a negative result. Do **not** swap prompts, κ, or
hyperparameters after looking at held-out numbers. That is the exact failure in the CS682 report
(the "best" row there was chosen by test IoU).

## Figures and tables (all produced by `analyze.py` / `viz.py`, per dataset)

Suggested paper layout: Fig 1 = controlled dose-response (the clean mechanism), Fig 2 = success vs
static top-1 and hacking across the 4 datasets, Fig 3 = COCO-Search18 gaze agreement plus CGL, and
Fig 4 = trajectories on Waldo. The per-dataset outputs are:

1. **Fig 1** `fig_success_hacking.pdf`: per reward, held-out success (RL vs exhaustive argmax of the
   reward) and hacked rate. This is the headline: the gap between the two success markers is
   optimisation, and the hacked rate is misspecification.
2. **Fig 2** `fig_budget.pdf`: anytime success vs glimpse budget. The gaze policy uses tens of glimpses
   where exhaustive scanning uses 1024.
3. **Fig 3** `fig_trajectories.pdf`: fixation paths of `clip_full` vs `clip_conj` / `learned_pess` on the
   same held-out images (pick images from `summary.json` *before* looking at the paths: e.g. the first
   test image of each fold, which is what `viz.py` does by default).
4. **Table 1** `table_main.tex`: main numbers with CIs. **Table 2**: the single-prompt diagnostic
   (replaces the report's test-selected query table).

## Related work to cite (do not skip)

- Active vision / glimpse agents: Mnih et al. 2014 (RAM), Ba et al. 2015 (DRAM), Caicedo & Lazebnik
  2015 (active object localisation with RL).
- Reward misspecification: Amodei et al. 2016; Skalse et al. 2022 (defining reward hacking);
  Gao et al. 2023 (reward-model overoptimisation scaling).
- Uncertainty-penalised / conservative rewards: Hadfield-Menell et al. 2017 (Inverse Reward Design);
  Coste et al. 2024 (reward-model ensembles mitigate overoptimisation).
- CLIP compositionality failures: Yuksekgonul et al. 2023 (bag-of-words behaviour, ARO).
- Gaze-guided learning: Zhang et al. 2018 (AGIL); Saran et al. 2021 (coverage-based gaze loss).
  Only as motivation and future work; this project uses no human gaze.
- Shaping: Ng, Harada & Russell 1999.

Verify every citation's year and venue before submitting.

## Known limitations to state up front

- Small dataset (tens of scenes). The CIs will be wide; the paper is about the *phenomenon* and the
  measurement protocol, not a SOTA number.
- One encoder (CLIP ViT-B/32) and a fixed glimpse grid. Waldo can straddle cells; the two-way
  pointing criterion mitigates this but does not remove it.
- `hacked` is defined relative to each reward's own score, so it is comparable across seeds and folds
  of one reward, and only loosely comparable across rewards.
