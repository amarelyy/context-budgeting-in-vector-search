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

### 2026-11-08: The four checks on TEST (Phase 8)
- Config: test (75); conf_score = 5-signal logistic model (T = 349); main method = global conformal budget q_hat 1,649, alpha 0.1, method A; bootstrap 95% CIs (2,000 resamples, seed 0); ECE 5 bins; LLM main setting for fixed k = 11, 12 added. Files: `results/phase8_checks.json`, `results/plots/reliability_test.png`, `results/llm_fixed_k_11_12_test_main_seed42.csv`. Details: `results/runs/21_four_checks_test.md`.
- Numbers: auroc 0.677 [0.535, 0.806] vs top1_dist 0.623 [0.488, 0.757], diff CI [−0.112, +0.217]. ece 0.038 [0.018, 0.147]. coverage (conditional) 0.931 [0.862, 0.983], unconditional 0.720 (ceiling 0.773). tokens_saved_pct at equal recall vs fixed-k: global −9.8%, bins3 −7.3%, CQR −1.9%; vs k=20: 41.1%. answer_f1: global 0.521, fixed k=12 0.537, paired diff −0.016 [−0.058, +0.021].
- Why it changed: first run of the four checks. Check 4 fails.

### 2026-11-08: Token-budget retrieval on TEST (Phase 7)
- Config: test split (75), pre-declared main = global conformal budget q_hat 1,649 (alpha 0.1, method A); secondary bins3 and CQR (5 signals). LLM main setting (temp 0, seed 42, num_ctx 5120). Added after the run: fixed-k with k chosen on calibration by conformal on gold_pos_rank (k = 12). Files: `results/budget_test_main_seed42.csv`, `results/fixed_k_curve_test.csv`. Details: `results/runs/20_budget_retrieval_test.md`.
- Numbers:

| method | recall_at_budget (both) | cond. coverage | total_tokens mean | saved vs k=20 | EM | answer_f1 |
|---|---|---|---|---|---|---|
| global conformal (main) | 0.720 | 0.931 | 1,551.2 | 41.1% | 0.400 | 0.521 |
| bins3 | 0.733 | 0.948 | 1,561.7 | 40.7% | 0.453 | 0.548 |
| CQR | 0.720 | 0.931 | 1,439.0 | 45.4% | 0.413 | 0.523 |
| fixed k=12 (calibration-chosen) | 0.760 | 0.983 | 1,542.5 | 41.4% | | |
| fixed k=11 (hindsight) | 0.720 | 0.931 | 1,412.8 | 46.4% | | |

- Why it changed: first test run of the method. Coverage target met; no gain over a calibration-chosen fixed-k.

### 2026-11-08: Phase 6 options 2 + 3: CQR and question-type signal (dev estimate, no test)
- Config: method A, alpha 0.1, 425 fit+calibration queries, out-of-fold predictions, 500 re-splits (75 / 350). CQR = linear quantile regression (q 0.9, no penalty) on log tokens + one conformal correction. Question type = TF-IDF + logistic regression on fit questions. Files: `ablations/phase6_methods/`. Details: runs 18, 19.
- Numbers: tokens fed at ~0.91-0.92 conditional coverage: global 1,399; 5 signals: bins3 1,614, CQR 1,505; 6 signals: bins3 1,608, CQR 1,556. Score AUROC (OOF, fit) 0.609 → 0.704 with the question-type signal. Question-type classifier AUROC 0.954. Best case with true types and no penalty: 1,320 vs 1,356 global (−2.7%).
- Why it changed: tried CQR (option 2), then added the question-type signal (option 3). Neither beats the global budget at alpha 0.1.

### 2026-11-08: Conformal budget per bin (Phase 6) + dev estimate (no test)
- Config: alpha 0.1, method A, 3 bins (edges = fit conf_score tertiles 0.3423 / 0.4191). Dev estimate: 425 fit+calibration queries, out-of-fold scores, 500 re-splits (75 / 350). Files: `data/processed/conformal.json`, `ablations/phase6_mondrian/`. Details: `results/runs/17_conformal_mondrian.md`.
- Numbers: real calibration q_hat global 1,649; bins 2,097 / 1,731 / 950. In-sample coverage 0.965 (bins) vs 0.930 (global). Dev estimate: tokens fed global 1,399 (45.1% saved), 2 bins 1,507 (40.9%), 3 bins 1,614 (36.7%); conditional coverage 0.908 / 0.914 / 0.923. Fixed k=12: 0.929, 1,486 tokens.
- Why it changed: first per-bin budget. Negative result: bins cost tokens (small-sample penalty per bin + weak score).

### 2026-11-08: Score model (Phase 5, fit only)
- Config: target y = gold_pos_tokens ≤ T, T = 349 (fit median), uncoverable → 0. Logistic regression, StandardScaler, C = 1.0. 5-fold stratified CV on fit, 5 split seeds. Details: `results/runs/16_score_model.md`.
- Numbers: CV AUROC all 5 signals 0.589 ± 0.010; rival top1_dist 0.460 ± 0.031; margin_mc alone 0.555. conf_score on fit 0.155-0.690 (median 0.381).
- Why it changed: first score model.

### 2026-11-08: Phase 6 ablation, coverage handling × alpha (fit + calibration only)
- Config: 425 pooled fit+calibration queries, 500 random re-splits (75 cal / 350 eval, seed 0), one global split-conformal budget, no signals. Methods A (conditional) and C (infinite); alpha 0.05-0.50. Files: `ablations/phase6_coverage/`.
- Numbers: A, alpha 0.1: conditional coverage 0.908, unconditional 0.693, 1,399 tokens, saved 45.1% vs k=20. C, alpha 0.1: q_hat = ∞ in 100% of re-splits, saved 0%. Fixed k=11 on the same queries: conditional coverage 0.920, 1,355 tokens.
- Why it changed: first Phase 6 ablation. Shows a global token budget ≈ fixed-k, so the gain must come from per-query signals.

### 2026-11-08: Labels (Phase 4, fit + calibration summary)
- Config: both-gold rule, top_m 20, Qwen3 tokenizer. Uncoverable queries kept with gold_pos_tokens = NaN. File: `data/processed/labels.csv`. Details: `results/runs/13_labels.md`.
- Numbers: fit 267 / 350 coverable (0.763), gold_pos_tokens mean 591.5, median 349.0, p90 1312.8. Calibration 57 / 75 coverable (0.760), mean 579.5, median 366.0, p90 1401.4. gold_pos_rank median 3 (fit), 4 (calibration).
- Why it changed: first labels. Fit numbers match the oracle (run 04) exactly.

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
