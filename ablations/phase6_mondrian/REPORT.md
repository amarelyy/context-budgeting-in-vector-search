# Phase 6 dev estimate: per-bin (Mondrian) budget vs one global budget

**Date:** 2026-11-08 · **Data:** fit + calibration only (no test data) · **Script:** `run_mondrian.py`

## Question
Phase 6 gives each `conf_score` bin its own budget `q_hat`. Does that use fewer tokens than one global budget at the same coverage? This is the step where the signals should start paying off.

## Setup
- Method A (conditional coverage), alpha = 0.1, finite-sample correction.
- `conf_score` for fit queries is **out-of-fold** (5-fold), so no query is scored by a model that saw it.
- Fit + calibration pooled (425), **500 random re-splits**: 75 to compute `q_hat`, 350 to evaluate.
- Bin edges = equal-count quantiles of `conf_score` on the non-calibration rows, so edges and `q_hat` use different data.
- n_bins = 1 (global), 2, 3. Fixed-k on the same 425 queries as reference.
- Outputs: `summary.csv`, `raw_repeats.csv`.

## Results

| n_bins | cond. coverage (mean / p05) | uncond. coverage | tokens fed | saved vs k=20 (mean ± std) | coverable queries in smallest bin (median) |
|---|---|---|---|---|---|
| 1 (global) | 0.908 / 0.838 | 0.693 | **1,399** | **45.1% ± 12.0** | 57 |
| 2 | 0.914 / 0.836 | 0.698 | 1,507 | 40.9% ± 11.3 | 24 |
| 3 | 0.923 / 0.861 | 0.704 | 1,614 | 36.7% ± 10.8 | 14 |

- Paired over the same re-splits: 2 bins use **+107** tokens vs global (cheaper in only 24% of re-splits); 3 bins use **+215** (cheaper in 16%).
- 3 bins: in 2.6% of re-splits a bin was too small and got `q_hat` = ∞ (feed all 20).

Fixed-k reference:

| k | cond. coverage | uncond. coverage | tokens fed | saved |
|---|---|---|---|---|
| 10 | 0.895 | 0.682 | 1,221 | 52.1% |
| 11 | 0.920 | 0.701 | 1,355 | 46.9% |
| 12 | 0.929 | 0.708 | 1,486 | 41.7% |

Real calibration split (`src/conformal.py`, in-sample): global `q_hat` 1,649; bins low / mid / high = 2,097 / 1,731 / 950 (coverable n = 17 / 20 / 20, median tokens needed 645 / 399 / 288). Mean budget 1,607 with bins vs 1,649 global.

## Findings

### Good
1. **The score does sort queries.** On the real calibration split, the median tokens needed falls from 645 (low confidence) to 399 (mid) to 288 (high), and the high-confidence bin gets a much smaller budget (950 vs 2,097).
2. **Coverage holds.** Every variant reaches the 0.90 conditional target on average.

### Negative (the main result)
3. **Bins cost tokens instead of saving them.** More bins means more tokens: global 1,399 → 2 bins 1,507 → 3 bins 1,614.
4. **The planned method does not beat fixed-k on dev data.** 3 bins (0.923 coverage, 1,614 tokens) uses more tokens than fixed k = 12 (0.929, 1,486).

### Why
5. **Each bin pays its own small-sample penalty.** The finite-sample correction takes the ⌈(n+1)(1−alpha)⌉-th value. With 57 coverable calibration queries in one global pool this is a moderate quantile. Split into bins of ~14-24, it becomes the bin's largest or second-largest value, so each bin over-covers (0.923 vs a 0.90 target) and wastes tokens.
6. **The score is too weak to make up for it.** With AUROC ≈ 0.59 (run 16), the bins differ only moderately. The token saving in the high-confidence bin is smaller than the extra tokens the small-sample penalty adds in every bin.

## What this means
The project's core idea (signals → per-query budget) **does not pay off with the current setup**: weak distance-only signals plus only 75 calibration queries. This is not a code problem; the pieces work as designed. Per CLAUDE.md, this should be discussed with Princess before going on to Phase 7.

## Options (not done; Princess decides)
| Option | What changes | Why it might help | Cost |
|---|---|---|---|
| **1. More data** | Larger HotpotQA subset (e.g. 2,000 queries → ~300 calibration) | Removes most of the small-sample penalty, which is the main cause found here | Rerun Phases 1-6 (fast) and the LLM baseline (~2 min per setting) |
| **2. Conformalized quantile regression (CQR)** instead of bins | Predict each query's needed tokens from the signals, then conformalize with one shared correction | No per-bin penalty; uses all calibration queries at once. Still conformal, still component 2 | New code in Phase 6 |
| **3. Stronger signal** | Add a question-type signal (comparison vs bridge) | The current signals are weak (AUROC 0.59) | Plan change; must stay inside component 2 |

Suggested order: 1 first (cheapest and addresses the main cause), then 2. Report the current negative result either way; it is a real finding.

## Limits
- Dev estimate on fit + calibration, not test. The 500 re-splits overlap, so they are not independent.
- Bin edges are equal-count quantiles; other edge choices were not tried.
