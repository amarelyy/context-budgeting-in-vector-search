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

### 2026-11-08: Index check (Phase 1, fit split only)
- Config: HotpotQA distractor validation, 500 queries, seed 42, split 350/75/75, fit split used (350 queries) / all-MiniLM-L6-v2, L2-normalized / HNSW M 32, efConstruction 200, efSearch 100 / 4956 chunks
- Numbers: HNSW vs exact recall@20 = 0.9994 (min 0.95); top-1 identical = 1.0. Both gold in top 5 / 10 / 20 = 0.520 / 0.683 / 0.763; at least one gold = 0.971 / 0.991 / 0.994. Top-1 distance median 0.8411 (min 0.3733, max 1.117).
- Why it changed: first run. This is NOT the baseline (the baseline is on the test split in Phase 2).
