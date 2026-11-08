# Run 08: Signal distributions (tasks 3.3-3.4)

## Goal
Compute the signals for all 500 queries, save them, and look at their distributions for anything strange.

## Setup
- `python src\signals.py` saves `data/processed/signals.csv` (500 rows) and `data/processed/retrieval_top20.json`.
- `python notebooks\plot_signals.py` plots **fit and calibration only**, so test is not looked at (looking at test could bias choices like `tau`).
- Plots: `results/plots/signals_hist.png`, `results/plots/margin_multihop_fit.png`.

## What happened
- 500 rows (350 fit, 75 calibration, 75 test), 0 missing values.
- Means, fit vs calibration: `margin_mc` 0.121 vs 0.124, `var_local` 0.0078 vs 0.0080, `top1_dist` 0.826 vs 0.824, `elbow_idx` 1.97 vs 1.99, `entropy` 0.736 vs 0.735.
- Spearman correlation on fit:

| | margin_mc | var_local | top1_dist | elbow_idx | entropy |
|---|---|---|---|---|---|
| margin_mc | 1.00 | 0.40 | -0.63 | -0.73 | -0.69 |
| var_local | | 1.00 | -0.88 | 0.02 | -0.90 |
| top1_dist | | | 1.00 | 0.23 | 0.91 |
| elbow_idx | | | | 1.00 | 0.24 |

- Multi-hop check on fit:

| | n | margin_mc median |
|---|---|---|
| both gold at ranks 1-2 | 95 | 0.076 |
| other queries | 255 | 0.101 |

## Good findings
- **Fit and calibration look alike.** Conformal calibration needs this similarity.
- **`entropy` with `tau` 0.1 spreads well** (0.13 to 0.99), so `tau` stays.

## Surprising
- **`margin_mc` is smaller when retrieval is perfect.** Confirmed on all 95 fit queries with both gold at ranks 1-2. A signal designed for single-answer search behaves differently on multi-hop data. This is a good point for the failure analysis (9.3) and the report.
- **The proposal's thresholds would not work here.** `var_local` median is 0.006, while the proposal's `theta_var` is 0.05. This supports replacing hand thresholds with conformal calibration.
- **Only one near-duplicate pair in the corpus.** One fit query had `margin_mc` exactly 0: two "Colomac Mine" chunks of the same length with almost the same text. It is the only such pair among 4,956 chunks.

## Stagnant / weak
- **`var_local`, `top1_dist` and `entropy` are almost the same signal** (|correlation| 0.88-0.91). When the best match is far away, the neighbours are bunched and the scores are flat. Expect unstable logistic regression weights (5.1) and small changes when one of them is removed (9.1).
- **`elbow_idx` barely moves.** It is 1 for more than half the queries (median 1). It may add little.

## What went wrong
- Nothing broke.

## Next
- 4.1: decide whether "covered" means both gold or at least one. The margin finding matters for this.
- 9.4 (dedup) will likely change little on HotpotQA. BEIR may be different.
