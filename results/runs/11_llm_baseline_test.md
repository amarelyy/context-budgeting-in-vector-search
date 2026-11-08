# Run 11: Fixed-k baseline with the LLM (task 2.4)

## Goal
Add answer quality to the fixed-k baseline: feed the top k chunks to the LLM and score its answer against the HotpotQA answer. Also run k = 0 (no retrieval) to see how much the LLM answers from memory.

## Setup
- Test split, 75 queries, k = 0, 5, 10, 20, so 300 LLM calls.
- Model `qwen3:4b-instruct-2507-q4_K_M`, Ollama 0.40.15. **Main setting:** temperature 0, seed 42, num_ctx 5120, num_predict 256.
- Prompt: `src/prompt.py`. Truncation guard: `llm.chat_checked`.
- Metrics: EM and F1 with the official HotpotQA normalization (`src/qa_metrics.py`, self-check passes).
- Per-query CSV: `results/llm_baseline_test_main_seed42.csv` (300 rows).
- Command: `python src\llm_baseline.py main`. Took 780 s.

## What happened

| k | recall (both gold) | tokens fed (mean) | EM | F1 |
|---|---|---|---|---|
| 0 | 0.000 | 0 | 0.160 | 0.217 |
| 5 | 0.560 | 595.2 | 0.400 | 0.518 |
| 10 | 0.693 | 1274.1 | 0.413 | 0.537 |
| 20 | 0.773 | 2634.0 | 0.440 | 0.546 |

- Truncated: 0 of 300. Cut off by num_predict: 0 of 300. `ollama ps` during the run: CONTEXT 5120, 100% GPU, 3.4 GB.
- Answers are short: median 2 words at k = 20 (max 15).

F1 split by whether both gold chunks were fed:

| k | both gold fed: n / mean F1 | not both: n / mean F1 |
|---|---|---|
| 5 | 42 / 0.688 | 33 / 0.302 |
| 10 | 52 / 0.635 | 23 / 0.314 |
| 20 | 58 / 0.629 | 17 / 0.263 |

Per query, k = 5 → k = 20: F1 rises for 12 queries, drops for 8, stays the same for 55.

## Good findings
- **More context barely helps the answer.** From k = 5 to k = 20, tokens grow 4.4× (+2,039) but F1 rises only +0.028 (0.518 → 0.546). This is the strongest evidence so far that fixed-k wastes tokens, and now it is measured on answers, not only on recall.
- **Retrieval clearly matters.** With both gold fed, F1 is about 0.63-0.69. Without both, it is about 0.26-0.31. So `recall_at_budget` is a meaningful target for answer quality.
- **The pipeline is clean.** No truncation, no cut-off answers, short answers as asked.

## Surprising
- **The LLM knows some answers without any context.** At k = 0, 12 of 75 queries are exactly right (F1 = 1) and 20 have F1 > 0. HotpotQA comes from Wikipedia, so the model has seen some of it. This is why the k = 0 row matters.
- **More chunks can hurt.** For 8 queries, F1 drops from k = 5 to k = 20. Also, among queries where both gold were fed, mean F1 falls from 0.688 (k = 5) to 0.629 (k = 20). Caution: these groups contain different queries, so this is a hint of distraction or "lost in the middle", **not proof**. Task 9.3 tests it properly.

## Stagnant / weak
- **15 queries at k = 20 have both gold fed but F1 = 0.** These are LLM failures, not retrieval failures. Examples:
  - wrong entity: "Estonia" vs gold "Belarus", "Jenn Brown" vs "Kristine Leahy"
  - wrong yes/no: "No" vs "yes"
  - a refusal: "None of the passages mention a university in Pullman, Washington..." vs "Edward R. Murrow"
  - close but scored 0 by strict F1: "genera of flowering plants" vs "genus"

  A 4B model is weak at 2-hop reasoning (Risk 1), so `answer_f1` stays the secondary metric.
- F1 overall is modest (~0.55 at best). This is the LLM's limit, not something budgeting can fix.

## What went wrong
- **Slow calls because of `localhost`.** Every call took ~2.5 s, even k = 0 with a 60-token prompt. On Windows, `localhost` tries IPv6 first and falls back after ~2.2 s. Measured: 2.4 s with `localhost` vs 0.15 s with `127.0.0.1`. Fixed in `src/llm.py`. Answers are not affected (temperature 0), so the run was not repeated. The `latency_s` column in this CSV is inflated by ~2.2 s per row and should not be used.
- **Run exceeded the 10-minute foreground limit** and moved to the background. Only one run happened.
- **Token count off by one for one query.** In 3 of 300 rows (one query at k = 5, 10, 20), Ollama counted 1 more token than we did. That is the safe direction (nothing dropped), so it is not truncation. Cause not checked; probably one special character tokenized differently.

## Next
- Secondary setting (Qwen's values, several seeds) for error analysis: not run yet. The number of seeds needs Princess's decision.
- These numbers are the baseline that Phase 7 must beat on the same 75 test queries.
