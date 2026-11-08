# Run 16: Score model (Phase 5, tasks 5.1-5.3)

## Goal
Turn the five signals into one `conf_score` per query: the chance that the query is "easy" (both gold passages within a small token budget). Phase 6 will give each `conf_score` bin its own budget, so easy queries get a smaller budget.

## Setup
- **Target (5.1):** y = 1 if `gold_pos_tokens` ≤ T, else 0. Uncoverable queries → 0. **T = 349 tokens** = median `gold_pos_tokens` of coverable fit queries (starting value). Fit positives: 134 of 350 (0.383).
- **Model:** logistic regression (StandardScaler → LogisticRegression, L2, C = 1.0).
- **Rival (5.2):** the same model on `top1_dist` only.
- **Judged on fit only:** 5-fold stratified cross-validation, repeated with 5 split seeds. Calibration and test are only scored, never used to fit or choose anything.
- **Output (5.3):** `data/processed/conf_scores.csv` (500 rows: `conf_score`, `conf_score_top1`, `y`), `score_model.joblib`, `score_model_top1.joblib`, `score_model_meta.json`.
- Command: `python src\score_model.py`.

## What happened

5-fold CV AUROC on fit (mean ± std over 5 split seeds):

| model | AUROC |
|---|---|
| **all 5 signals (main)** | **0.589 ± 0.010** |
| top1_dist only (rival) | 0.460 ± 0.031 |
| margin_mc only | 0.555 ± 0.005 |
| var_local only | 0.527 ± 0.008 |
| elbow_idx only | 0.515 ± 0.027 |
| entropy only | 0.448 ± 0.015 |

Sensitivity to the target (all 5 signals, seed 42, info only):

| target | positives | AUROC |
|---|---|---|
| T = p25 = 199 tokens | 0.197 | 0.616 |
| T = p50 = 349 tokens (chosen) | 0.383 | 0.609 |
| T = p75 = 778 tokens | 0.571 | 0.644 |
| gold_in_pool | 0.763 | 0.699 |

Main model coefficients (standardized): margin_mc −0.852, var_local −0.313, top1_dist −0.245, elbow_idx −0.102, entropy −0.648.

`conf_score` on fit: min 0.155, median 0.381, max 0.690.

## Good findings
- **Combining the signals helps a little.** 0.589 for all 5, versus 0.555 for the best single signal.
- **The main model beats the rival** (0.589 vs 0.460). Formally this passes the shape of check 1 ("above 0.5 and above `top1_dist`"), but only on fit. The real check is 8.1 on test.
- **The model learned the multi-hop effect by itself.** `margin_mc` has a **negative** weight (−0.852): a larger margin means *less* likely easy. This matches run 08 (perfect 2-hop retrieval gives a small margin).

## Surprising
- **`top1_dist` alone is worse than a coin flip in CV (0.460).** It carries no usable information for this target, so the fitted direction flips between folds. In the UARP proposal it was the natural rival; here it is not a real competitor.
- **The signals predict "is the gold in the pool at all" better (0.699) than "is the query easy" (0.609).** Distances say more about whether retrieval found both passages than about how deep the second one sits.

## Stagnant / weak
- **0.589 is weak.** The plan asks for "clearly above 0.5"; this is only somewhat above.
- **`conf_score` is squeezed into 0.155-0.690.** The model is rarely confident either way. In Phase 6 the bins will therefore look similar to each other, and their budgets (`q_hat`) may differ only a little from the single global budget. Expect only a small gain over the global baseline (run 14).
- Coefficients for `var_local`, `top1_dist` and `entropy` should not be read one by one: these signals are correlated at about 0.9 (run 08).

## What went wrong
- Nothing broke.
- Note: T = median was decided before looking at the results. p75 gave a higher AUROC (0.644), but switching now would be choosing T for the best number. It is kept as recorded, and the other T values can be tried in the ablation (9.1).

## Next
- Phase 6: Mondrian conformal, one `q_hat` per `conf_score` bin, on the calibration split (alpha 0.1, method A).
- If the bins do not beat the global budget, discuss the question-type signal idea with Princess (run 15).
