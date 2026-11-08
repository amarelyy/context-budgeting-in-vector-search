# Phase 6 methods: global vs bins vs CQR, with and without a question-type signal

**Date:** 2026-11-08 · **Data:** fit + calibration only (no test data) · **Scripts:** `run_methods.py`, `src/cqr.py`, `src/question_type.py`

## Question
Per-bin budgets lost to one global budget (`ablations/phase6_mondrian/`). Princess chose to try:
- **Option 2:** conformalized quantile regression (CQR), which gives a per-query budget without splitting calibration into small bins.
- **Option 3:** a stronger signal, the question type (comparison vs bridge).

## Setup
- Method A (conditional coverage), alpha 0.1. Fit + calibration pooled (425), **500 random re-splits** (75 calibrate / 350 evaluate). All model outputs for fit rows are **out-of-fold**.
- **Global:** one conformal `q_hat`.
- **Bins3:** 3 `conf_score` bins (logistic score), one `q_hat` each.
- **CQR:** linear quantile regression (quantile 0.9, no penalty) of log(`gold_pos_tokens`) on the signals, fitted on coverable fit rows. One conformal correction Q from all coverable calibration rows. Budget = exp(prediction + Q).
- **Question-type signal `q_comparison`:** TF-IDF (word 1-2 grams) + logistic regression trained on fit questions to predict HotpotQA's "comparison" label. Out-of-fold on fit; trained on all fit for the other splits. True labels come from the cached HotpotQA download; `data/raw` is unchanged.
- Outputs: `summary_base.csv`, `summary_qtype.csv`, `raw_repeats_base.csv`, `raw_repeats_qtype.csv`.

## Results

### Question-type signal (fit)
- Classifier (out-of-fold): AUROC 0.954, accuracy 0.857. Comparison share: fit 0.194, calibration 0.133.
- By true type on fit:

| type | n | coverable | median tokens needed | p90 tokens needed |
|---|---|---|---|---|
| bridge | 282 | 0.730 | 437.5 | 1,498 |
| comparison | 68 | 0.897 | 181.0 | 694 |

- Spearman(`q_comparison`, `gold_pos_tokens`) on coverable fit: **−0.321** (the five distance signals: all |ρ| ≤ 0.08).
- Score model (logistic, target y = covered within 349 tokens), out-of-fold AUROC on fit: **0.609 (5 signals) → 0.704 (6 signals)**.

### Dev estimate (500 re-splits)

| features | method | cond. coverage (mean / p05) | uncond. coverage | tokens fed | saved vs k=20 | paired vs global |
|---|---|---|---|---|---|---|
| 5 signals | **global** | 0.908 / 0.838 | 0.693 | **1,399** | **45.1%** | |
| 5 signals | bins3 | 0.923 / 0.861 | 0.704 | 1,614 | 36.7% | +215 (cheaper in 16%) |
| 5 signals | CQR | 0.908 / 0.833 | 0.693 | 1,505 | 41.0% | +106 (cheaper in 27%) |
| 6 signals | bins3 | 0.923 / 0.855 | 0.704 | 1,608 | 36.9% | +209 (cheaper in 21%) |
| 6 signals | CQR | 0.908 / 0.828 | 0.693 | 1,556 | 39.0% | +157 (cheaper in 25%) |
| | fixed k = 11 | 0.920 | 0.701 | 1,355 | 46.9% | |

(Global does not use any signal, so it is the same in both feature sets.)

### Real calibration split (in-sample)
- CQR, 5 signals: Q = +0.143, budgets 1,028-2,412 (mean 1,503), conditional coverage 0.912.
- CQR, 6 signals: Q = +0.108, budgets 655-2,277 (mean 1,536), conditional coverage 0.930. `q_comparison` weight −0.200 (comparison → smaller budget, as expected).

### Best case for the question-type idea
Coverable fit + calibration (324), true types, **no** finite-sample penalty: the 90% budget is 1,467 for bridge and 795 for comparison (global 1,356). A perfect per-type budget averages 1,320, **only 2.7% less than global.**

## Findings

### Good
1. **CQR fixes the small-bin problem.** It keeps coverage exactly on target (0.908, like global) instead of over-covering like bins (0.923), and it beats bins in tokens (1,505 vs 1,614).
2. **The question-type signal is real and strong.** Comparison questions need far fewer tokens (median 181 vs 437). It is easy to detect from the question text (AUROC 0.954), and it lifts the score model from 0.609 to 0.704 AUROC.

### Negative (the main result)
3. **No per-query method beats the single global budget at 90% coverage**, with or without the new signal. Every variant uses 100-215 more tokens than global on average, and each is cheaper in only 16-27% of re-splits.

### Why (the key insight)
4. **At a 90% coverage target, the budget is set by the hardest ~10% of queries, and those are bridge questions that no signal can tell apart.** Bridge questions have a p90 of 1,467 tokens versus 1,356 for all queries together. The question-type signal only makes the easy minority (comparison, ~22%) cheaper. Even a perfect per-type budget with unlimited calibration data saves only 2.7%.
5. **A weak predictor makes per-query budgets worse than one budget.** If budgets vary for reasons only loosely tied to the real need, some queries get too little and others too much. To still reach 90% coverage, the average budget must go up. Adding the question-type signal made CQR budgets vary more (655-2,277) without enough extra accuracy in the tail, so CQR got slightly worse (1,505 → 1,556).

### Surprising
6. **A better score (AUROC 0.70) did not mean a better budget.** AUROC measures sorting around the median (y = within 349 tokens), but the 90% budget depends on the tail. The score model was trained for the wrong part of the distribution for this purpose.

## What this means for the project
The original plan (per-query conformal budget from cheap signals beats fixed-k) **is not supported on HotpotQA at alpha = 0.1** with these signals and this data size. What holds:
- A **global conformal token budget** is as efficient as the best fixed-k (≈ k = 11) **and** chooses its size from data with a stated coverage guarantee, which fixed-k cannot.
- **The negative result is explained**, not just observed: the tail of hard bridge queries drives the budget, and distance signals carry almost no information about it.

## Options for Princess (not done)
| Option | Idea | Expected effect |
|---|---|---|
| A. Report as is | Global conformal budget + the explained negative result as the main finding | Honest, complete, ready for Phase 7-8 now |
| B. Lower coverage targets | Check whether per-query budgets help at alpha 0.3-0.5, where the median (not the tail) matters and comparison vs bridge differs a lot | May show a real gain in a lower-coverage regime; must be decided before test |
| C. More data | Larger subset (more calibration) | Removes small-sample noise, but the 2.7% best case shows the ceiling stays low at alpha 0.1 |

## Limits
- Dev estimate only (fit + calibration). Re-splits overlap.
- Linear quantile regression only; a non-linear model was not tried.
- `q_comparison` is learned from HotpotQA's own labels, so it does not transfer to BEIR as is.
