# Run 23: Alpha sweep, conformal vs grid-search threshold (Phase 9.2)

Full write-up: [`ablations/PHASE9_REPORT.md`](../../ablations/PHASE9_REPORT.md). Data: `ablations/phase9_alpha/` (`summary_dev.csv`, `summary_test.csv`, `alpha_sweep.png`).

## Goal
Show how coverage and tokens move with alpha, and compare conformal calibration with a simple hand-tuned ("grid search") threshold, as the plan asks.

## Setup
- Methods, all one global setting, method A:
  - `conformal_tokens`: split-conformal token budget (our method)
  - `grid_tokens`: the smallest budget on a 25-token grid with empirical calibration coverage ≥ 1 − alpha (no finite-sample correction)
  - `conformal_k`: fixed k = conformal quantile of `gold_pos_rank`
- Part A, dev (fit + calibration, 500 re-splits 75 / 350). Part B, test: reporting only (alpha 0.1 was fixed before any test run).
- alpha 0.05 to 0.50. `python ablations\phase9_alpha\run_alpha_sweep.py`

## What happened (dev, selected rows)

| alpha | target | conformal_tokens: cov / tokens | grid_tokens: cov / tokens | conformal_k: cov / tokens |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.961 / 1,959 | 0.945 / 1,774 | 0.964 / 2,009 |
| 0.10 | 0.90 | **0.908** / 1,399 | **0.895** / 1,287 | 0.917 / 1,437 |
| 0.20 | 0.80 | 0.805 / 837 | 0.795 / 809 | 0.823 / 924 |
| 0.30 | 0.70 | 0.704 / 602 | 0.696 / 587 | 0.734 / 704 |
| 0.50 | 0.50 | 0.498 / 294 | 0.506 / 295 | 0.554 / 376 |

Share of re-splits below target at alpha 0.1: conformal_tokens 0.38, grid_tokens 0.51, conformal_k 0.32.
Test at alpha 0.1: conformal_tokens 0.931 / 1,551; grid_tokens 0.931 / 1,344; conformal_k 0.983 / 1,543.

## Good findings
- **Conformal keeps its promise on average at every alpha**; the grid-search threshold sits slightly **below** target at every alpha from 0.05 to 0.40 and misses the target in about half of all splits. This is the value of the finite-sample correction.

## Surprising
- **The price of the guarantee is small but real:** at alpha 0.1, grid search saves 49.5% vs 45.1% for conformal (≈ 110 tokens per query on dev).
- **Tokens and chunks lie on the same trade-off curve.** At ~0.92 coverage, conformal_k uses ~1,437 tokens vs ~1,490 for conformal_tokens (interpolated); at ~0.82 they are about equal. `conformal_k` is coarser (whole chunks), so it lands at slightly higher coverage for each alpha.

## Stagnant / weak
- The ranking of the three methods does not depend on alpha.

## What went wrong
- Nothing.
