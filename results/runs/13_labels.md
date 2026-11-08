# Run 13: Labels (Phase 4, tasks 4.1-4.3)

## Goal
Compute, for every query, the smallest token budget that covers both gold passages (`gold_pos_tokens`). This is the label that the score model (Phase 5) and the conformal budget (Phase 6) learn from.

## Setup
- **4.1, multi-hop rule (Princess's choice):** "covered" = **both** gold chunks are inside the fed set.
  - `gold_pos_tokens` = sum of `token_cost` down the ranked list, up to and including the **last** gold chunk.
  - `gold_pos_rank` = the rank of that last gold chunk.
- **4.2, gold not in the top 20:** `gold_in_pool = False`, and `gold_pos_tokens` is left empty ("infinite"). These queries are kept, never dropped.
- Extra analysis columns: `first_gold_tokens` (tokens up to the first gold) and `pool_tokens` (k = 20 cost).
- Pool: the same top 20 as the signals (`data/processed/retrieval_top20.json`). Qwen3 tokenizer.
- Output: `data/processed/labels.csv`, 500 rows (fit 350, calibration 75, test 75).
- Command: `python src\labels.py`. The summary shows fit and calibration only; test labels are saved for evaluation but not looked at.

## What happened

| | fit | calibration |
|---|---|---|
| queries | 350 | 75 |
| both gold in top 20 (coverable) | 267 (0.763) | 57 (0.760) |
| only 1 gold in top 20 | 81 | 18 |
| 0 gold in top 20 | 2 | 0 |
| `gold_pos_tokens` mean (coverable) | 591.5 | 579.5 |
| `gold_pos_tokens` median | 349.0 | 366.0 |
| `gold_pos_tokens` p90 | 1312.8 | 1401.4 |
| `gold_pos_tokens` max | 3766 | 2097 |
| `gold_pos_rank` median | 3 | 4 |
| covered by rank 2 | 0.356 | 0.316 |
| covered by rank 5 | 0.682 | 0.632 |
| k = 20 pool tokens, mean | 2558.4 | 2516.7 |

## Good findings
- **The labels agree with the oracle from run 04 exactly** (fit: 267 coverable, mean 591.5 tokens). Two separate pieces of code give the same answer, so the label code is very likely correct.
- **Fit and calibration look alike again** (coverable 0.763 vs 0.760, median 349 vs 366). Good for conformal calibration.
- **The budget needed varies a lot between queries.** The median is 349 tokens, but the p90 is about 1,300-1,400. One fixed k cannot serve both ends, which is the case for a per-query budget.

## Surprising
- **The p90 budget (~1,300-1,400 tokens) is only about half of k = 20 (~2,550 tokens).** If 90% of coverable queries fit in about 1,400 tokens, even a single global budget (no signals) might save a lot. This gives us a simple extra baseline to check in Phase 7: "one fixed token budget for all queries".
- Only 2 of 350 fit queries have no gold at all in the top 20. Almost all misses are the **second** gold (81 of 83).

## Stagnant / weak
- **24% of queries can never be covered** within the top 20. No budget, signal or calibration can fix them; only better retrieval can (out of scope).

## What went wrong
- Nothing broke.
- **A design problem was found for Phase 6.** With alpha = 0.1, the plan asks for about 90% coverage. Because 24% of queries can never be covered, unconditional coverage can be at most about 76%, so 90% is impossible. Options are listed in TASK.md (Open questions). Suggested: calibrate on coverable queries only (conditional coverage), and report unconditional coverage next to the 76% ceiling.

## Next
- 5.1: the score model needs a binary target, for example "both gold covered within a token position T". T must be chosen on fit.
- Phase 6: Princess decides how to handle the uncoverable 24%.
