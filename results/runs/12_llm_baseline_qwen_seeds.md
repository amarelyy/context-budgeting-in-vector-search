# Run 12: Fixed-k baseline with the LLM, Qwen's sampling values, 3 seeds (task 2.4, secondary setting)

## Goal
Measure how much answer quality moves by chance when the LLM samples (temperature 0.7), and find which errors are stable. This is for error analysis only. Headline numbers stay with the main setting (run 11).

## Setup
- Same as run 11 (test split, 75 queries, k = 0, 5, 10, 20, same prompt, num_ctx 5120, num_predict 256).
- **Secondary setting:** temperature 0.7, top_p 0.8, top_k 20, min_p 0 (Qwen's recommended values). Seeds 1, 2, 3 (3 is a starting value).
- Per-query CSVs: `results/llm_baseline_test_qwen_seed1.csv`, `..._seed2.csv`, `..._seed3.csv`.
- Command: `python src\llm_baseline.py qwen 1 2 3`. Took about 105 s per seed.

## What happened

Mean ± std over the 3 seeds, with the main setting (run 11) for comparison:

| k | EM (Qwen, 3 seeds) | F1 (Qwen, 3 seeds) | F1 range | F1 main (temp 0) |
|---|---|---|---|---|
| 0 | 0.142 ± 0.008 | 0.216 ± 0.010 | 0.209-0.227 | 0.217 |
| 5 | 0.400 ± 0.000 | 0.520 ± 0.002 | 0.518-0.522 | 0.518 |
| 10 | 0.413 ± 0.013 | 0.522 ± 0.023 | 0.496-0.538 | 0.537 |
| 20 | 0.427 ± 0.013 | 0.542 ± 0.012 | 0.529-0.551 | 0.546 |

- Truncated 0, cut by num_predict 0, in all 900 calls.
- Stability per query (4 runs: main + 3 Qwen seeds):

| | F1 = 0 in all 4 runs | F1 > 0 in all 4 runs | changes between runs |
|---|---|---|---|
| k = 5 | 27 | 46 | 2 |
| k = 20 | 24 | 46 | 5 |

- Identical answer text across the 3 Qwen seeds: 60 of 75 queries at k = 5, 53 of 75 at k = 20.
- The 24 always-wrong queries at k = 20: 13 had both gold fed (LLM failure), 11 did not (retrieval missed a gold).

## Good findings
- **The main setting is representative.** Main-setting F1 sits inside the Qwen seed range at every k. Temperature 0 does not give a lucky or unlucky number.
- **Errors are stable.** Only 2-5 of 75 queries change between right and wrong across runs. The errors come from the questions themselves (retrieval or LLM), not from sampling luck. This makes error analysis (9.3) reliable.
- **The speed fix worked.** About 105 s per 300 calls (mean 0.36 s per call), compared with 780 s before the `127.0.0.1` fix.

## Surprising
- **The F1 gain from more context is about the size of the noise.** From k = 5 to k = 20, F1 rises +0.022 (Qwen mean) or +0.028 (main). Seed-to-seed spread alone is up to ±0.023 (k = 10). So with 75 test queries we **cannot** say that k = 20 gives better answers than k = 5. This makes finding 1 even stronger: the extra tokens buy no measurable answer quality.
- k = 10 is the noisiest (F1 range 0.496-0.538), and with seed 1 it scores *below* k = 5.
- More context makes answers vary more: identical answers drop from 60 (k = 5) to 53 (k = 20) of 75.

## Stagnant / weak
- k = 5 EM is exactly 0.400 in all 3 seeds. With short contexts, the sampling barely matters.
- About a third of queries (24-27) are wrong in every run. Neither more context nor sampling fixes them.

## What went wrong
- Nothing broke.
- Limitation: 3 seeds and 75 queries is small. The std values are rough.

## Next
- When comparing the token-budget method with fixed-k (Phase 8), treat F1 differences smaller than about ±0.02 as noise. `recall_at_budget` stays the primary metric.
- The 13 "both gold fed, always wrong" queries are the LLM failure set for task 9.3.
