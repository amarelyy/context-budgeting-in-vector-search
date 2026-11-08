# Run 20: Token-budget retrieval on test (Phase 7, tasks 7.1-7.3)

## Goal
First run of the method on the test split, compared with fixed-k on the same 75 queries.

## Setup
- **Pre-declared before the run (Princess's choice "langsung aja global budget"):** MAIN = global conformal budget, `q_hat` = 1,649 tokens (from calibration, Phase 6). Secondary, reported but never chosen from: 3 bins (`q_hat` 2,097 / 1,731 / 950) and CQR (5 signals, Q = +0.143).
- Budget rule: add chunks in ranked order while the running total of `token_cost` ≤ budget.
- LLM main setting: temperature 0, seed 42, num_ctx 5120, same prompt as the baseline.
- `python src\budget_retrieval.py` → `results/budget_test_main_seed42.csv` (225 rows), `results/fixed_k_curve_test.csv`. 225 LLM calls in 197 s.
- **Baseline added after the test run:** fixed-k with k chosen on calibration by the same conformal rule applied to `gold_pos_rank` (alpha 0.1, 57 coverable calibration queries) → k = 12. This is a *stronger* rival, so adding it after the fact can only make our method look worse, not better.

## What happened
Test: 75 queries, both gold in the top 20 for 58 (0.773).

| method | recall (both gold) | cond. coverage (target 0.90) | tokens fed | saved vs k=20 | avg chunks | EM | F1 |
|---|---|---|---|---|---|---|---|
| **global conformal (main)** | 0.720 | **0.931** | 1,551.2 | 41.1% | 12.63 | 0.400 | 0.521 |
| bins3 (secondary) | 0.733 | 0.948 | 1,561.7 | 40.7% | 12.72 | 0.453 | 0.548 |
| CQR (secondary) | 0.720 | 0.931 | 1,439.0 | 45.4% | 11.80 | 0.413 | 0.523 |
| fixed k = 12 (chosen on calibration) | 0.760 | 0.983 | 1,542.5 | 41.4% | 12 | | |
| fixed k = 11 (hindsight: smallest k with recall 0.720 on test) | 0.720 | 0.931 | 1,412.8 | 46.4% | 11 | | |
| fixed k = 10 (baseline, run 11) | 0.693 | 0.897 | 1,274.1 | 51.6% | 10 | 0.413 | 0.537 |
| fixed k = 20 (baseline, run 11) | 0.773 | 1.000 | 2,634.0 | 0% | 20 | 0.440 | 0.546 |

0 truncated calls.

## Good findings
- **The coverage guarantee holds on test.** Conditional coverage is 0.931 against a 0.90 target, using a budget chosen only from 75 calibration queries.
- **41% fewer tokens than k = 20.** Answer F1 is 0.521 vs 0.546, a −0.025 difference, about the size of the seed noise measured in run 12 (±0.02).

## Surprising
- **CQR did better on test than global** (1,439 vs 1,551 tokens at the same recall), the opposite of the dev estimate (run 18). With one test split of 75 queries this is within noise. It was pre-declared as secondary and is not promoted.

## Stagnant / weak (the main result)
- **The token budget does not beat a fairly chosen fixed-k.** Fixed-k with k picked on calibration by the same conformal rule (k = 12) uses about the same tokens (1,543 vs 1,551) and covers more (0.983 vs 0.931).
- At equal recall (0.720), the hindsight best fixed-k (k = 11) uses 9.8% fewer tokens than the global budget. That k could not have been known without looking at test.
- So on this data the useful part of the method is the **conformal choice of size** (it works for ranks as well as for tokens). Counting in **tokens** rather than chunks adds nothing, consistent with the dev estimates (runs 14, 17-19).

## What went wrong
- Nothing broke.
- Honesty note: the calibration-chosen fixed-k baseline was not in the original plan and was added after seeing test. It favours fixed-k, not our method.

## Next
- Phase 8: the four checks (auroc on test vs `top1_dist`, ece + reliability diagram, coverage vs 1 − alpha, `tokens_saved_pct` at equal recall).
