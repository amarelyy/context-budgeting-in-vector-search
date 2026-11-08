# Results log

Every result gets one entry. Newest first. Do not log a number that was not run.

## Entry format

```
### YYYY-MM-DD: <short title>
- Config: k / alpha / bins / seed / split / dataset / embedding model / tokenizer
- Numbers: <metric = value, ...>
- Why it changed: <one line, compared with the previous run>
```

## Entries

### 2026-11-08: Fixed-k baseline with LLM, secondary setting (Qwen's values, 3 seeds, test split)
- Config: same as the main-setting entry below, but temperature 0.7, top_p 0.8, top_k 20, min_p 0, seeds 1/2/3. CSVs: `results/llm_baseline_test_qwen_seed{1,2,3}.csv`. Details: `results/runs/12_llm_baseline_qwen_seeds.md`. For error analysis only, not headline numbers.
- Numbers (mean ± std over 3 seeds):

| k | EM | answer_f1 | F1 range |
|---|---|---|---|
| 0 | 0.142 ± 0.008 | 0.216 ± 0.010 | 0.209-0.227 |
| 5 | 0.400 ± 0.000 | 0.520 ± 0.002 | 0.518-0.522 |
| 10 | 0.413 ± 0.013 | 0.522 ± 0.023 | 0.496-0.538 |
| 20 | 0.427 ± 0.013 | 0.542 ± 0.012 | 0.529-0.551 |

  Per query (main + 3 seeds): at k = 20, 24 always wrong, 46 always right, 5 flip.
- Why it changed: only the sampling setting changed. The main-setting F1 lies inside the seed range at every k. The k = 5 → 20 F1 gain (~+0.02) is about the size of seed noise.

### 2026-11-08: Fixed-k baseline with LLM (task 2.4, test split, main setting)
- Config: test split (75 queries), k = 0/5/10/20, qwen3:4b-instruct-2507-q4_K_M (Ollama 0.40.15), temperature 0, seed 42, num_ctx 5120, num_predict 256, prompt `src/prompt.py`, EM/F1 with HotpotQA normalization. Per-query CSV: `results/llm_baseline_test_main_seed42.csv`. Details: `results/runs/11_llm_baseline_test.md`.
- Numbers:

| k | recall_at_k (both gold) | total_tokens mean | EM | answer_f1 |
|---|---|---|---|---|
| 0 | 0.000 | 0 | 0.160 | 0.217 |
| 5 | 0.560 | 595.2 | 0.400 | 0.518 |
| 10 | 0.693 | 1274.1 | 0.413 | 0.537 |
| 20 | 0.773 | 2634.0 | 0.440 | 0.546 |

  Truncated 0/300, cut by num_predict 0/300.
- Why it changed: first LLM run. Adds answer quality to the retrieval baseline. k = 5 → 20 is 4.4× the tokens for +0.028 F1.

### 2026-11-08: Oracle upper bound (FIT split only, NOT a method result)
- Config: fit split (350 queries), top 20 candidates, token_cost tokenizer Qwen/Qwen2.5-7B-Instruct, other settings as the baseline. Oracle = stop exactly when the gold is covered (uses labels, impossible in practice). Rule: oracle is computed on fit only, never on test. Script: `notebooks/oracle_fit.py`.
- Numbers: both gold in top 20 for 267 of 350 queries. Oracle tokens to cover both gold: mean 591.5, median 349.0. To cover first gold: mean 143.2, median 96.0. k=20 mean tokens (all fit queries) = 2558.4. **Maximum possible saving vs k=20 (both gold, same 267 queries) = 76.7%.**
- Why it changed: replaces an earlier oracle computed on test (deleted by rr's rule; nothing was tuned from it).

### 2026-11-08: Fixed-k baseline, rerun with Qwen tokenizer (test split)
- Config: as the first baseline below, but token_cost tokenizer = Qwen/Qwen2.5-7B-Instruct (was all-MiniLM-L6-v2).
- Numbers:

| k | recall_at_k (both gold) | recall_at_k (any gold) | gold_frac | total_tokens mean | total_tokens median |
|---|---|---|---|---|---|
| 5 | 0.560 | 0.947 | 0.753 | 595.2 | 575.0 |
| 10 | 0.693 | 0.960 | 0.827 | 1274.1 | 1261.0 |
| 20 | 0.773 | 0.973 | 0.873 | 2634.0 | 2573.0 |

- Why it changed: only the tokenizer changed (decision 2.3: use an LLM, so count tokens like the LLM does). Recall is identical (same retrieval); tokens are about 10% higher because Qwen splits text into more tokens. **This is the baseline to use from now on.**

### 2026-11-08: Fixed-k baseline (Phase 2, test split)
- Config: HotpotQA distractor validation, 500 queries, seed 42, split 350/75/75, **test split (75 queries)** / all-MiniLM-L6-v2, L2-normalized / HNSW M 32, efConstruction 200, efSearch 100 / 4956 chunks / token_cost tokenizer: all-MiniLM-L6-v2 / k = 5, 10, 20
- Numbers:

| k | recall_at_k (both gold) | recall_at_k (any gold) | gold_frac | total_tokens mean | total_tokens median |
|---|---|---|---|---|---|
| 5 | 0.560 | 0.947 | 0.753 | 539.0 | 519.0 |
| 10 | 0.693 | 0.960 | 0.827 | 1153.1 | 1148.0 |
| 20 | 0.773 | 0.973 | 0.873 | 2379.2 | 2337.0 |

- Why it changed: first baseline run. Going from k=10 to k=20 doubles tokens (+1226) but adds only +0.080 "both gold" recall. Retrieval-only metrics (2.3 not decided yet). Ranked top-20 lists saved to `data/processed/baseline_test_top20.json`.

### 2026-11-08: Index check (Phase 1, fit split only)
- Config: HotpotQA distractor validation, 500 queries, seed 42, split 350/75/75, fit split used (350 queries) / all-MiniLM-L6-v2, L2-normalized / HNSW M 32, efConstruction 200, efSearch 100 / 4956 chunks
- Numbers: HNSW vs exact recall@20 = 0.9994 (min 0.95); top-1 identical = 1.0. Both gold in top 5 / 10 / 20 = 0.520 / 0.683 / 0.763; at least one gold = 0.971 / 0.991 / 0.994. Top-1 distance median 0.8411 (min 0.3733, max 1.117).
- Why it changed: first run. This is NOT the baseline (the baseline is on the test split in Phase 2).
