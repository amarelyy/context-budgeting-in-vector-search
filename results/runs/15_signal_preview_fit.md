# Run 15: Signal preview, do the signals know how many tokens a query needs? (fit only)

## Goal
Princess asked why the global budget is far from the oracle ceiling. The gap can only close if the signals can tell, before retrieval is finished, which queries need few tokens. This run is a quick preview on the fit split. It is **not** the formal check 1 (8.1), which uses the fitted score model on test.

## Setup
- Fit split only. Signals from `signals.csv`, labels from `labels.csv`.
- (a) Spearman correlation of each signal with `gold_pos_tokens` (267 coverable fit queries).
- (b) Single-signal AUROC for two binary targets on all 350 fit queries: `gold_in_pool`, and "both gold in the top 2". Reported as max(AUROC, 1 − AUROC), so the direction does not matter.
- Commands: inline Python (pandas, `sklearn.metrics.roc_auc_score`).

## What happened
- `gold_pos_tokens` (coverable fit): median 349, mean 592, p75 778, p90 1313. 50% of queries need ≤ 350 tokens, 67% need ≤ 600.
- With the global `q_hat` of 1447 (A, alpha 0.1), the median query receives about **1,098 more tokens than it needs**.
- Spearman with `gold_pos_tokens`: margin_mc +0.00, var_local −0.06, top1_dist +0.07, elbow_idx −0.03, entropy +0.08.
- Split by `top1_dist` thirds (89 queries each): median tokens needed 360 / 317 / 517, p90 1482 / 851 / 1820.
- Single-signal AUROC:

| signal | gold_in_pool | both gold in top 2 |
|---|---|---|
| margin_mc | 0.67 | 0.58 |
| var_local | 0.55 | 0.50 |
| top1_dist | 0.53 | 0.52 |
| elbow_idx | 0.64 | 0.58 |
| entropy | 0.55 | 0.51 |

## Good findings
- `margin_mc` (0.67) and `elbow_idx` (0.64) carry some information about whether both gold passages are in the pool at all.

## Surprising
- **The signals barely relate to how many tokens a query needs** (|Spearman| ≤ 0.08). The distances describe how well the question matches its best passages. They say little about where the second (bridge) gold passage sits, which is what sets the budget.

## Stagnant / weak
- For "both gold in the top 2" (an easy query), the best single signal reaches only 0.58 AUROC. Combining signals in a model (Phase 5) may help a little, but this is a **warning that check 1 (8.1) may fail** with the current signals.

## What went wrong
- Nothing broke. This is a preview, not a formal check.

## Next
- Phase 5 fits the combined model on fit. If it is also weak, discuss with Princess before going further (the 8.1 GATE rule).
- Possible extra signal (not added, needs Princess's approval): HotpotQA question type (comparison vs bridge). Comparison questions name both entities, so both gold passages are easy to retrieve.
