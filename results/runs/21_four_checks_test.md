# Run 21: The four checks on test (Phase 8, tasks 8.1-8.4)

## Goal
Run the four checks from CONTEXT.md on the test split: does the score predict, is it calibrated, is coverage right, does it pay off.

## Setup
- Test split, 75 queries. `python src\evaluate.py` → `results/phase8_checks.json`, `results/plots/reliability_test.png`, `results/llm_fixed_k_11_12_test_main_seed42.csv`.
- Bootstrap 95% CIs: 2,000 resamples of the test queries, seed 0 (n is small).
- 8.1-8.2 use `conf_score` (5-signal logistic model, target y = both gold within 349 tokens). 8.3-8.4 use the main method (global conformal budget, run 20).
- 8.4 adds LLM runs for fixed k = 11 and 12 (main setting, 150 calls, 0 truncated) so F1 is compared at matched context sizes.
- "Fixed-k tokens at equal recall" = linear interpolation on the test fixed-k curve (k = 1..20).

## What happened

| check | metric | result (95% CI) | hoped for | verdict |
|---|---|---|---|---|
| 1. Does the score predict? | `auroc` | **0.677** [0.535, 0.806]; rival `top1_dist` 0.623 [0.488, 0.757]; difference [−0.112, +0.217] | clearly > 0.5 and > `top1_dist` | **weak pass**: above 0.5, above the rival on the point estimate, but the difference is not significant |
| 2. Is it calibrated? | `ece` (5 bins) | **0.038** [0.018, 0.147]; mean conf 0.372 vs observed rate 0.360 | close to 0 | **pass, but trivial**: 52 of 75 queries fall in one bin (conf ≈ 0.32), so the score is calibrated but barely spread out |
| 3. Is coverage right? | `coverage` (conditional, global budget) | **0.931** [0.862, 0.983] on 58 coverable; unconditional 0.720 (ceiling 0.773) | close to 1 − alpha = 0.90 | **pass** |
| 4. Does it pay off? | `tokens_saved_pct` at equal `recall_at_budget` vs fixed-k | **−9.8%** (global: 1,551 tokens vs 1,413 for fixed-k at recall 0.720); bins3 −7.3%; CQR −1.9% | positive | **fail** |

Reliability bins (test): n = 1 / 52 / 20 / 2, mean conf 0.137 / 0.322 / 0.487 / 0.639, observed 0.000 / 0.288 / 0.500 / 1.000.

Answer F1 (main setting): fixed k = 5 / 10 / 11 / 12 / 20 → 0.518 / 0.537 / 0.520 / 0.537 / 0.546; global budget 0.521, bins3 0.548, CQR 0.523. Paired F1, global − fixed k = 12: −0.016 [−0.058, +0.021].

## Good findings
- **The coverage guarantee works out of sample** (check 3): 0.931 vs a 0.90 target, from 75 calibration queries.
- **The score generalizes better than its fit CV suggested:** test AUROC 0.677 vs 0.589 in CV on fit. With 75 queries the CI is wide, so this is probably partly luck.
- **Against k = 20, the method saves 41% of tokens** with no significant F1 change.

## Surprising
- **The rival `top1_dist` scores 0.623 on test**, though it was 0.460 in fit CV. Single-signal results move a lot on small samples.
- **Low ECE is not a sign of a good score here.** The score mostly predicts the base rate (≈ 0.36), and a score that always says the base rate is perfectly calibrated but useless. The reliability diagram shows 52 of 75 queries in one bin.

## Stagnant / weak
- **Check 4 fails.** At equal recall, fixed-k uses fewer tokens than every conformal variant. The token-budget idea does not beat fixed-k on HotpotQA with these signals, matching the dev estimates (runs 14, 17-19) and run 20.
- Check 1 passes only weakly: the score beats the rival on the point estimate, but the CI of the difference includes 0.
- F1 differences between all methods are within noise.

## What went wrong
- Nothing broke. 0 truncated LLM calls.
- Note: "fixed-k tokens at equal recall" uses the test curve, so fixed-k gets the benefit of hindsight in check 4 (as the plan defines it). The calibration-chosen fixed-k (k = 12, run 20) is the fair version and gives the same conclusion.

## Next
- Phase 9 (ablation, failure analysis, dedup, BEIR transfer) and Phase 10 (wrap-up). Princess decides the scope given these results.
