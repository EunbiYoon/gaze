# Results

Images: 713. Unit = image (seed-averaged); 95% bootstrap CI over images.

**WARNING - incomplete grid, numbers below are NOT final:**
- oracle: 2/15 runs, 286/713 images
- clip_full: 2/15 runs, 286/713 images
- clip_full_cgl: 2/15 runs, 286/713 images

## Main table (held-out images, greedy policy from all level-0 starts)

| reward | RL success | hacked | distractor | near-miss | timeout | steps | train success | static top-1 | coarse-to-fine | seed std | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 43.5 [37.9, 49.4] | 0.0 [0.0, 0.0] | 23.6 [18.9, 28.5] | 21.7 [17.1, 26.6] | 11.2 [7.7, 15.0] | 13.8 | 51.4 [48.1, 54.7] | 100.0 [100.0, 100.0] | 99.7 [99.0, 100.0] | nan | - |
| clip_full | 50.4 [44.6, 55.9] | 5.2 [2.8, 7.9] | 21.0 [16.5, 25.7] | 21.2 [16.6, 26.0] | 7.4 [4.6, 10.5] | 9.7 | 53.1 [49.7, 56.4] | 75.2 [69.9, 80.1] | 16.1 [11.9, 20.6] | nan | - |
| clip_full_cgl | 41.6 [36.0, 47.2] | 5.1 [2.7, 7.8] | 26.1 [21.3, 31.1] | 25.7 [21.0, 30.7] | 6.6 [3.9, 9.6] | 11.4 | 43.6 [40.2, 47.1] | 75.2 [70.3, 80.1] | 16.1 [11.9, 20.3] | nan | - |

hacked = stopped on a non-target glimpse that the TRAINING reward scores >= the best target glimpse.
distractor = wrong stop >1 cell from the target; near-miss = wrong stop adjacent to it.

## Paired comparisons (A - B, held-out success, per image)

| A | B | mean diff (pp) | 95% CI | Wilcoxon p | A>B / A<B / tie |
|---|---|---|---|---|---|
| clip_full_cgl | clip_full | -8.8 | [-14.9, -2.7] | 0.0163 | 47 / 76 / 163 |
| clip_full | oracle | +6.8 | [+1.0, +12.6] | 0.137 | 60 / 45 / 181 |

No multiple-comparison correction is applied; with 2 comparisons, treat p > 0.025 (Bonferroni) as inconclusive.

## Single-prompt diagnostic (no RL, all images)

| prompt | top-1 (target is argmax) | target percentile | median # distractors above target |
|---|---|---|---|
| full | 70.8 [67.3, 74.1] | 99.0 [98.7, 99.3] | 0 |
| name | 70.8 [67.5, 74.2] | 99.0 [98.7, 99.3] | 0 |
| para1 | 67.7 [64.4, 71.1] | 98.7 [98.3, 99.1] | 0 |
| para2 | 66.9 [63.4, 70.4] | 98.6 [98.3, 99.0] | 0 |
| para5 | 65.8 [62.1, 69.3] | 98.1 [97.6, 98.6] | 0 |
| para3 | 64.1 [60.6, 67.6] | 98.6 [98.2, 98.9] | 0 |
| para4 | 61.9 [58.2, 65.4] | 97.9 [97.4, 98.4] | 0 |

## Agreement with human search fixations (held-out images)

Pearson CC between the agent's pooled fixation map and the human fixation map on the peripheral grid. Reference: a centre-bias map scores 0.158 [0.149, 0.167].

| reward | gaze CC | success |
|---|---|---|
| oracle | 0.324 [0.284, 0.365] | 43.5 [37.9, 49.4] |
| clip_full | 0.349 [0.307, 0.393] | 50.4 [44.6, 55.9] |
| clip_full_cgl | 0.330 [0.289, 0.372] | 41.6 [36.0, 47.2] |
