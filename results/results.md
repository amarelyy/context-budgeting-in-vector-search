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
