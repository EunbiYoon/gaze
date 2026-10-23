# Results

Images: 160. Unit = image (seed-averaged); 95% bootstrap CI over images.

**WARNING - incomplete grid, numbers below are NOT final:**
- oracle: 2/5 runs, 64/160 images
- clip_name: 2/5 runs, 64/160 images
- clip_full: 2/5 runs, 64/160 images
- clip_mean_attr: 2/5 runs, 64/160 images
- clip_conj: 2/5 runs, 64/160 images
- clip_pe_mean: 2/5 runs, 64/160 images
- clip_pe_pess: 2/5 runs, 64/160 images
- learned: 2/5 runs, 64/160 images
- learned_pess: 2/5 runs, 64/160 images

## Main table (held-out images, greedy policy from all level-0 starts)

| reward | RL success | hacked | distractor | near-miss | timeout | steps | train success | static top-1 | coarse-to-fine | seed std | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.3 [0.0, 0.7] | 0.0 [0.0, 0.0] | 96.4 [95.6, 97.2] | 3.3 [2.5, 4.1] | 0.0 [0.0, 0.0] | 3.0 | 0.4 [0.2, 0.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | nan | - |
| clip_name | 1.8 [0.5, 3.4] | 40.2 [30.5, 50.1] | 91.4 [87.6, 94.8] | 6.5 [3.2, 10.4] | 0.3 [0.0, 0.8] | 4.7 | 2.2 [1.1, 3.6] | 15.6 [7.8, 25.0] | 1.6 [0.0, 4.7] | nan | - |
| clip_full | 1.8 [0.1, 4.9] | 28.3 [18.6, 38.6] | 72.0 [62.1, 81.3] | 18.8 [10.5, 27.7] | 7.5 [2.5, 13.5] | 6.2 | 2.2 [0.6, 4.1] | 17.2 [7.8, 26.6] | 1.6 [0.0, 4.7] | nan | - |
| clip_mean_attr | 2.5 [0.3, 5.4] | 31.2 [22.6, 39.8] | 83.2 [76.7, 89.1] | 9.0 [4.5, 14.3] | 5.3 [1.6, 9.7] | 6.1 | 2.1 [1.0, 3.5] | 4.7 [0.0, 10.9] | 0.0 [0.0, 0.0] | nan | - |
| clip_conj | 0.4 [0.1, 0.8] | 52.0 [43.3, 60.8] | 92.2 [87.9, 95.8] | 6.6 [3.2, 10.8] | 0.8 [0.0, 2.3] | 5.3 | 0.5 [0.2, 1.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | nan | - |
| clip_pe_mean | 0.2 [0.0, 0.5] | 28.8 [19.0, 39.0] | 78.1 [69.5, 86.4] | 16.2 [8.9, 24.1] | 5.5 [1.3, 10.6] | 6.6 | 3.0 [1.0, 5.5] | 18.8 [9.4, 28.1] | 3.1 [0.0, 7.8] | nan | 0 |
| clip_pe_pess | 3.0 [0.1, 7.5] | 28.3 [18.7, 38.7] | 80.7 [71.9, 88.7] | 8.8 [3.6, 15.0] | 7.5 [2.2, 13.8] | 7.0 | 1.4 [0.4, 2.7] | 17.2 [7.8, 26.6] | 3.1 [0.0, 7.8] | nan | 0.5 |
| learned | 2.9 [0.3, 6.7] | 15.6 [8.5, 23.5] | 75.9 [67.2, 84.4] | 10.8 [5.1, 17.7] | 10.4 [4.3, 17.2] | 7.8 | 3.2 [1.2, 5.7] | 20.3 [10.9, 29.7] | 7.8 [1.6, 14.1] | nan | 0 |
| learned_pess | 2.7 [0.1, 6.7] | 18.8 [10.9, 27.6] | 72.6 [62.7, 81.9] | 11.1 [5.1, 18.2] | 13.6 [6.2, 21.6] | 8.3 | 3.7 [1.7, 6.1] | 20.3 [10.9, 31.2] | 7.8 [1.6, 15.6] | nan | 0,0.5 |

hacked = stopped on a non-target glimpse that the TRAINING reward scores >= the best target glimpse.
distractor = wrong stop >1 cell from the target; near-miss = wrong stop adjacent to it.

## Paired comparisons (A - B, held-out success, per image)

| A | B | mean diff (pp) | 95% CI | Wilcoxon p | A>B / A<B / tie |
|---|---|---|---|---|---|
| clip_conj | clip_full | -1.4 | [-4.5, +0.4] | 0.994 | 2 / 2 / 60 |
| clip_conj | clip_mean_attr | -2.1 | [-5.0, +0.0] | 0.627 | 3 / 5 / 56 |
| clip_pe_pess | clip_pe_mean | +2.8 | [-0.1, +7.2] | 0.635 | 3 / 1 / 60 |
| learned_pess | learned | -0.2 | [-5.2, +4.8] | 0.538 | 4 / 7 / 53 |
| clip_full | oracle | +1.5 | [-0.4, +4.7] | 0.814 | 4 / 3 / 57 |

No multiple-comparison correction is applied; with 5 comparisons, treat p > 0.010 (Bonferroni) as inconclusive.

## Single-prompt diagnostic (no RL, all images)

| prompt | top-1 (target is argmax) | target percentile | median # distractors above target |
|---|---|---|---|
| para1 | 21.9 [15.6, 28.7] | 90.9 [88.6, 93.1] | 6 |
| para4 | 21.9 [15.6, 28.1] | 90.3 [87.8, 92.5] | 7 |
| full | 20.0 [13.8, 26.2] | 90.8 [88.5, 93.0] | 6 |
| para2 | 20.0 [13.8, 26.2] | 90.1 [87.6, 92.4] | 7 |
| para5 | 16.9 [11.2, 23.1] | 90.2 [87.9, 92.4] | 6 |
| para3 | 15.6 [10.0, 21.2] | 89.8 [87.5, 92.0] | 8 |
| name | 10.6 [6.2, 15.6] | 82.3 [79.0, 85.5] | 22 |

## Breakdown by `n_partial` (held-out success / hacked, %)

| reward | n_partial=0 | n_partial=4 | n_partial=16 | n_partial=48 |
|---|---|---|---|---|
| oracle | 0.7 / 0.0 (n=17) | 0.4 / 0.0 (n=17) | 0.0 / 0.0 (n=14) | 0.0 / 0.0 (n=16) |
| clip_name | 2.6 / 29.4 (n=17) | 0.4 / 30.1 (n=17) | 4.5 / 55.4 (n=14) | 0.0 / 49.2 (n=16) |
| clip_full | 0.4 / 18.0 (n=17) | 5.9 / 22.1 (n=17) | 0.4 / 47.3 (n=14) | 0.0 / 29.3 (n=16) |
| clip_mean_attr | 5.9 / 19.5 (n=17) | 0.4 / 32.0 (n=17) | 0.0 / 44.6 (n=14) | 3.5 / 30.9 (n=16) |
| clip_conj | 0.4 / 58.5 (n=17) | 0.0 / 38.2 (n=17) | 0.9 / 60.3 (n=14) | 0.4 / 52.3 (n=16) |
| clip_pe_mean | 0.4 / 25.0 (n=17) | 0.4 / 21.7 (n=17) | 0.0 / 40.6 (n=14) | 0.0 / 30.1 (n=16) |
| clip_pe_pess | 0.4 / 24.3 (n=17) | 10.7 / 13.2 (n=17) | 0.4 / 46.0 (n=14) | 0.0 / 33.2 (n=16) |
| learned | 5.5 / 7.0 (n=17) | 4.4 / 13.2 (n=17) | 0.9 / 19.6 (n=14) | 0.4 / 23.8 (n=16) |
| learned_pess | 5.9 / 5.5 (n=17) | 4.0 / 19.5 (n=17) | 0.4 / 22.8 (n=14) | 0.0 / 28.9 (n=16) |
