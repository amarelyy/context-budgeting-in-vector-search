# Run 03: Fixed-k baseline, retrieval only (tasks 2.1-2.2)

## Goal
Measure the standard RAG approach (always feed k chunks) on the test split, so every later method has something to beat on the same queries.

## Setup
- Test split, 75 queries. k = 5, 10, 20.
- Metrics:
  - recall (both gold): share of queries with both gold chunks inside the top k (strict)
  - recall (any gold): share of queries with at least one gold chunk inside
  - gold_frac: average share of gold found (0, 0.5 or 1 per query)
  - tokens fed: sum of `token_cost` of the k chunks
- Command: `python src\baseline.py`. Also saves the ranked lists to `data/processed/baseline_test_top20.json`.
- Using test here is allowed: fixed-k has nothing to tune. (Rule in CLAUDE.md updated to say this.)

## What happened
First run (MiniLM tokenizer):

| k | recall both | recall any | gold_frac | tokens mean | tokens median |
|---|---|---|---|---|---|
| 5 | 0.560 | 0.947 | 0.753 | 539.0 | 519.0 |
| 10 | 0.693 | 0.960 | 0.827 | 1153.1 | 1148.0 |
| 20 | 0.773 | 0.973 | 0.873 | 2379.2 | 2337.0 |

Rerun after switching to the Qwen tokenizer (the baseline to use from now on):

| k | recall both | recall any | gold_frac | tokens mean | tokens median |
|---|---|---|---|---|---|
| 5 | 0.560 | 0.947 | 0.753 | 595.2 | 575.0 |
| 10 | 0.693 | 0.960 | 0.827 | 1274.1 | 1261.0 |
| 20 | 0.773 | 0.973 | 0.873 | 2634.0 | 2573.0 |

## Good findings
- **Fixed-k wastes tokens at the high end.** From k = 10 to k = 20, tokens double (+1,360 with Qwen) but both-gold recall rises only +0.080. This is the waste the project wants to remove.
- **Test looks like fit.** Test both-gold recall (0.560 / 0.693 / 0.773) is close to fit (0.520 / 0.683 / 0.763). No sign of a strange test split.

## Surprising
- Recall did not change at all between the two tokenizers. This is expected (retrieval does not use `token_cost`), but it is a useful check that the rerun was clean.

## Stagnant / weak
- Recall (any gold) is almost flat: 0.947 → 0.960 → 0.973. Finding one gold is solved already at k = 5.
- Tokens grow almost linearly with k (about 120-130 tokens per chunk). Fixed-k gives no control over cost.

## What went wrong
- **Rule conflict found.** CLAUDE.md said "never touch test before Phase 7", but TASK 2.1 asked for the baseline on test. Princess decided to follow TASK.md. CLAUDE.md now says "never tune on test".

## Next
- Run 11 adds the LLM answers (EM, F1) to this table, plus k = 0.
