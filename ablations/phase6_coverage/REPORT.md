# Phase 6 ablation: coverage handling and alpha

**Date:** 2026-11-08 · **Data:** fit + calibration only (no test data) · **Script:** `run_ablation.py`

## Question
24% of queries can never be covered: their second gold passage is not in the top 20. With alpha = 0.1, the plan asks for 90% coverage, but the unconditional ceiling is 76.2%. How should Phase 6 handle these queries, and which alpha makes sense?

## Setup
- Fit + calibration pooled: 425 queries, 324 coverable (0.762).
- **500 random re-splits** (seed 0): 75 queries to compute `q_hat` (same size as the real calibration split), 350 to evaluate. Numbers below are means over the 500 re-splits.
- Split conformal with the finite-sample correction: `q_hat` = the ⌈(n+1)(1−alpha)⌉-th smallest calibration score, or infinity if that index is larger than n.
- **One global budget for all queries.** No signals or bins yet (that needs the Phase 5 score model).
- Fed set = the longest prefix of the ranked top 20 that fits in `q_hat`. Covered = `gold_pos_tokens` ≤ `q_hat`. Uncoverable queries are never covered.
- Methods:
  - **A, conditional:** `q_hat` from coverable calibration queries only. Target: P(covered | both gold in top 20) ≥ 1 − alpha.
  - **C, infinite:** uncoverable queries count as needing infinite tokens, and `q_hat` comes from all queries. Target: P(covered) ≥ 1 − alpha. If `q_hat` = infinity, feed all 20.
  - **B** (unconditional with a larger alpha) is C at alpha ≥ 0.25, so the sweep covers it.
- alpha ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}.

## Results

Full table: `summary.csv`. Every re-split: `raw_repeats.csv`. Fixed-k on the same 425 queries: `fixed_k_reference.csv`. Plot: `coverage_tokens_vs_alpha.png`.

| Method | alpha | target | q_hat median | q_hat = ∞ share | cond. coverage (mean / p05) | uncond. coverage | tokens fed | saved vs k=20 |
|---|---|---|---|---|---|---|---|---|
| A | 0.05 | 0.95 | 2080 | 0% | 0.961 / 0.907 | 0.733 | 1959 | 23.2% |
| A | **0.10** | 0.90 | 1447 | 0% | **0.908** / 0.838 | 0.693 | 1399 | **45.1%** |
| A | 0.15 | 0.85 | 1043 | 0% | 0.857 / 0.774 | 0.654 | 1042 | 59.1% |
| A | 0.20 | 0.80 | 905 | 0% | 0.805 / 0.704 | 0.614 | 837 | 67.2% |
| A | 0.25 | 0.75 | 795 | 0% | 0.752 / 0.648 | 0.574 | 707 | 72.3% |
| A | 0.30 | 0.70 | 678 | 0% | 0.704 / 0.583 | 0.538 | 602 | 76.4% |
| A | 0.40 | 0.60 | 523 | 0% | 0.600 / 0.487 | 0.458 | 421 | 83.5% |
| A | 0.50 | 0.50 | 356 | 0% | 0.498 / 0.378 | 0.380 | 294 | 88.5% |
| C | 0.05 | 0.95 | ∞ | 100% | 1.000 / 1.000 | 0.763 | 2551 | 0.0% |
| C | 0.10 | 0.90 | ∞ | 100% | 1.000 / 1.000 | 0.763 | 2551 | 0.0% |
| C | 0.15 | 0.85 | ∞ | 99.4% | 1.000 / 1.000 | 0.763 | 2549 | 0.1% |
| C | 0.20 | 0.80 | ∞ | 89.2% | 0.996 / 0.961 | 0.760 | 2492 | 2.3% |
| C | 0.25 | 0.75 | 2766 | 43.4% | 0.971 / 0.886 | 0.741 | 2153 | 15.6% |
| C | 0.30 | 0.70 | 1782 | 15.4% | 0.931 / 0.830 | **0.711** | 1707 | 33.1% |
| C | 0.40 | 0.60 | 871 | 0.2% | 0.794 / 0.670 | **0.606** | 841 | 67.0% |
| C | 0.50 | 0.50 | 578 | 0% | 0.653 / 0.517 | **0.498** | 510 | 80.0% |

Fixed-k on the same 425 queries (selected rows):

| k | cond. coverage | uncond. coverage | tokens fed | saved vs k=20 |
|---|---|---|---|---|
| 7 | 0.806 | 0.614 | 826 | 67.6% |
| 10 | 0.895 | 0.682 | 1221 | 52.1% |
| 11 | 0.920 | 0.701 | 1355 | 46.9% |
| 14 | 0.957 | 0.729 | 1752 | 31.3% |
| 20 | 1.000 | 0.762 | 2551 | 0% |

## Findings

### Good
1. **Conformal does what it promises.** Method A's average conditional coverage lands on the target at every alpha (for example 0.908 for a 0.90 target, 0.805 for 0.80). Method C lands on its unconditional target once alpha ≥ 0.3 (0.711 for 0.70, 0.606 for 0.60, 0.498 for 0.50).
2. **Method A keeps alpha = 0.1 meaningful.** It gives 90.8% conditional coverage and saves 45% of tokens versus k = 20.

### Surprising
3. **Method C gives zero savings for alpha ≤ 0.2.** It cannot reach 80-95% when 24% of queries are impossible, so `q_hat` becomes infinite and it feeds all 20 chunks. It is valid but useless at the planned alpha. Option B (C with alpha ≥ 0.25) only starts saving at alpha 0.3.
4. **With one global budget, A and C sit on the same trade-off curve.** At about 0.61 unconditional coverage, A (alpha 0.2) saves 67.2% and C (alpha 0.4) saves 67.0%. Both choose a single threshold, so the choice between A and C only changes **what alpha means and which guarantee we can state**, not efficiency. The difference will matter with per-bin budgets: under C, low-confidence bins would often get `q_hat` = ∞ and fall back to k = 20.

### Stagnant / weak, and the most important finding
5. **A single global token budget is no better than fixed-k.** At the same conditional coverage, fixed-k uses about the same tokens:
   - A, alpha 0.1: 0.908 coverage with 1,399 tokens, versus fixed k = 11: 0.920 with 1,355 tokens.
   - A, alpha 0.2: 0.805 with 837 tokens, versus k = 7: 0.806 with 826 tokens.

   **Counting tokens instead of chunks gives no gain by itself.** Fixed-k also has no guarantee and no rule for choosing k; conformal at least chooses the budget from data with a stated coverage. But for the project to beat fixed-k on efficiency, **the per-query signals (Phase 5) must do the work.** This is now the key test of the project.
6. **One re-split can miss the target noticeably.** With 75 calibration queries, the worst 5% of re-splits for A at alpha 0.1 get only 0.838 coverage. The guarantee holds on average, not for every split (Risk 4).

### What went wrong
7. **Bug found and fixed before reporting.** The first run counted uncoverable queries as "covered" whenever `q_hat` = ∞, because `inf <= inf` is `True` in Python. That showed an impossible 100% unconditional coverage for C. Fixed with `covered = in_pool & (score <= q_hat)`. Only C's unconditional column changed. All numbers above are from the fixed run.

## Recommendation
- **Use method A (conditional) with alpha = 0.1** for Phase 6. Always report unconditional coverage next to the 0.762 ceiling.
- **Add "one global token budget (A, alpha 0.1)" as a baseline in Phase 7.** The signal-based budget must beat both fixed-k and this global budget.
- Revisit C only if Princess wants an unconditional guarantee. It then needs alpha ≥ 0.3.

## Limits
- No signals or bins yet. Phase 6 proper (Mondrian bins on `conf_score`) can behave differently.
- Coverage is the "both gold in the fed set" proxy, not answer correctness.
- 425 pooled queries; the re-splits overlap, so the 500 repeats are not independent.
