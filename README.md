# Index-Level Context Budgeting in Vector Search

**Goal:** minimize the context budget (in tokens) that a vector database feeds to an LLM in a RAG system, without losing the passage that holds the answer.

Standard RAG feeds a fixed number of passages (k) for every query. This project instead stores a `token_cost` for every chunk in the index and picks the most relevant chunks that fit inside a **token budget**. The budget is chosen per query from cheap retrieval signals and calibrated from data with conformal prediction.

> **Status: work in progress (draft README, updated 2026-11-08).** The baseline (retrieval + LLM answers), the oracle upper bound and the confidence signals are done. The token-budget method itself is not built yet. Rows marked "not run yet" have no results.

## Progress

| Phase | What | Status |
|---|---|---|
| 0 | Setup (venv, requirements) | done |
| 1 | Data, chunks, `token_cost`, embeddings, HNSW index, index checks | done |
| 2 | Fixed-k baseline: retrieval metrics + LLM answers (EM, F1), both sampling settings | done |
| 3 | Confidence signals: code, unit test, saved for all splits, histograms | done |
| 4 | Labels (`gold_pos_tokens`) | not started |
| 5-7 | Score model, conformal budget, token-budget retrieval | not started |
| 8 | The four checks | not started |
| 9-10 | Ablation, failure analysis, dedup, wrap-up | not started |

## Key findings so far

1. **More context barely improves answers.** From k = 5 to k = 20, tokens fed grow 4.4× (595 → 2634) but answer F1 rises only +0.028 (0.518 → 0.546). That gain is about the size of the run-to-run noise measured over 3 sampling seeds (±0.02).
2. **There is a lot of room to save.** An oracle that stops exactly at the gold passages would use 76.7% fewer tokens than k = 20 (fit split). This is the ceiling: no real method can beat it.
3. **Retrieval drives answer quality.** With both gold passages fed, F1 is about 0.63-0.69; without both, about 0.26-0.31.
4. **`margin_mc` behaves differently on multi-hop data.** When retrieval is perfect (both gold at ranks 1-2), the margin is *smaller* (median 0.076 vs 0.101), because two equally good passages tie at the top.
5. **Three signals carry mostly the same information.** `var_local`, `top1_dist` and `entropy` are correlated at about 0.9.

Detailed notes for every run: [results/runs/](results/runs/README.md).

## Method (planned)

1. **Index:** FAISS HNSW over L2-normalized embeddings. Each chunk stores its `token_cost`.
2. **Signals** (free at retrieval time, from the top 20 retrieved distances; implemented and unit-tested):

   | Signal | Meaning | Formula |
   |---|---|---|
   | `margin_mc` | how clearly the top-1 result wins | d2 − d1 |
   | `var_local` | how spread out the neighbours are | (1/m) Σ (d_i − mean)² |
   | `top1_dist` | how close the best result is (rival signal) | d1 |
   | `elbow_idx` | where the distances jump sharply | chunks before the biggest gap |
   | `entropy` | how flat the scores are, 0 to 1 | entropy of softmax(−d / 0.1) / log(m) |

   `top_m = 20` and the softmax temperature 0.1 are starting values. Unit test: distances 0.20, 0.21, 0.90 give `margin_mc` = 0.01 and `var_local` = 0.1074.
3. **Confidence score:** a logistic regression maps the signals to `conf_score`, the probability that the gold passages are covered. The rival to beat is a model using `top1_dist` only.
4. **Conformal budget:** on a held-out calibration split, pick a token budget `q_hat` per `conf_score` bin, so that the gold passages are inside the budget with probability about 1 − alpha.
5. **Token-budget retrieval:** add chunks in ranked order until the budget is full.
6. **Add-on:** index-time deduplication of near-duplicate chunks.

## Testing method

**Data.** HotpotQA (distractor setting, validation set): 500 queries sampled with seed 42 and split into fit / calibration / test = 350 / 75 / 75. One paragraph is one chunk, giving 4,956 chunks. Every query has 2 gold paragraphs. A BEIR subset is planned as a second dataset.

**Retrieval setup.**

| Part | Choice |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384 dims, L2-normalized) |
| Index | FAISS `IndexHNSWFlat`, M 32, efConstruction 200, efSearch 100 (starting values) |
| Distance | L2 on unit vectors, range [0, 2], smaller = closer |
| `token_cost` tokenizer | `Qwen/Qwen3-4B-Instruct-2507` (the answering LLM's tokenizer) |

HNSW was checked against exact search: recall@20 = 0.9994 on the fit split.

**Answering LLM.** Qwen3-4B-Instruct-2507, run locally with Ollama (`qwen3:4b-instruct-2507-q4_K_M`, model ID `0edcdef34593`, Q4_K_M, Ollama 0.40.15), through the native `/api/chat` endpoint (`src/llm.py`). It was chosen for speed and fit (8 GB VRAM). The project measures how much context to retrieve, not LLM strength.

| Item | Choice | Evidence |
|---|---|---|
| Options | every option is sent explicitly in each call | the model's built-in temperature is 0.7, so 0 must be passed on purpose |
| `num_ctx` | **5120**, the same for every method | largest k = 20 prompt over all 500 queries = 4,599 tokens, + 256 for the answer, rounded up |
| Prompt template | `src/prompt.py`, identical for every method | passages fed as text only, so each passage costs exactly its `token_cost`; fixed overhead ~136 tokens at k = 20 |
| Truncation guard | `llm.chat_checked` compares our token count with Ollama's `prompt_eval_count` | our count matches Ollama's exactly; a too-long prompt gets an HTTP 400 error on Ollama 0.40.15 (no silent truncation) |
| Endpoint address | `127.0.0.1`, not `localhost` | `localhost` added ~2.2 s per call on Windows |

**Two sampling settings.**

| Setting | Values | Used for |
|---|---|---|
| Main | temperature 0, seed 42, `num_predict` 256 | every headline number and the baseline vs method comparison |
| Secondary | Qwen's recommended values (temperature 0.7, top_p 0.8, top_k 20, min_p 0), seeds 1, 2, 3 (starting value) | error analysis and run-to-run variance only |

The two settings are never mixed in one table.

**Metrics.**
- **Primary:** `recall_at_budget` (share of queries where both gold passages are inside the fed set) and tokens fed.
- **Secondary:** `answer_f1` and exact match (EM) against the HotpotQA answer, with the official HotpotQA normalization (`src/qa_metrics.py`). A 4B model is weak at multi-hop reasoning, so a low `answer_f1` may reflect the LLM, not retrieval.
- **No-retrieval baseline (k = 0):** measures how much the LLM answers from memory, since HotpotQA comes from Wikipedia.

**The four checks.**

| Check | Metric | Hoped-for result |
|---|---|---|
| 1. Does the score predict? | `auroc` | above 0.5 and above `top1_dist` alone |
| 2. Is it calibrated? | `ece` | close to 0 |
| 3. Is coverage right? | `coverage` | close to 1 − alpha |
| 4. Does it pay off? | `tokens_saved_pct` at equal `recall_at_budget` | positive versus fixed-k |

These are targets, not results.

**Rules.**
- Nothing is tuned on the test split.
- The oracle upper bound is computed on the fit split only.
- One thing changes between runs.
- Per-query logs are saved as CSV in `results/`. Every run is logged in [results/results.md](results/results.md) and explained in [results/runs/](results/runs/README.md).

## Results so far

**Fixed-k baseline (test split, 75 queries, main setting: temperature 0, seed 42):**

| Method | recall (both gold) | recall (any gold) | avg tokens fed | EM | answer_f1 |
|---|---|---|---|---|---|
| No retrieval, k = 0 | n/a | n/a | 0 | 0.160 | 0.217 |
| Fixed-k, k = 5 | 0.560 | 0.947 | 595.2 | 0.400 | 0.518 |
| Fixed-k, k = 10 | 0.693 | 0.960 | 1274.1 | 0.413 | 0.537 |
| Fixed-k, k = 20 | 0.773 | 0.973 | 2634.0 | 0.440 | 0.546 |
| Token-budget (conformal) | not run yet | not run yet | not run yet | not run yet | not run yet |

0 of 300 LLM calls were truncated or cut off. Per-query log: `results/llm_baseline_test_main_seed42.csv`.

**Secondary setting (Qwen's values: temperature 0.7, seeds 1, 2, 3; error analysis only):**

| k | EM (mean ± std) | answer_f1 (mean ± std) | F1 range over seeds |
|---|---|---|---|
| 0 | 0.142 ± 0.008 | 0.216 ± 0.010 | 0.209-0.227 |
| 5 | 0.400 ± 0.000 | 0.520 ± 0.002 | 0.518-0.522 |
| 10 | 0.413 ± 0.013 | 0.522 ± 0.023 | 0.496-0.538 |
| 20 | 0.427 ± 0.013 | 0.542 ± 0.012 | 0.529-0.551 |

- The main-setting F1 lies inside the seed range at every k, so temperature 0 gives a representative number.
- Errors are stable: at k = 20, 24 queries are wrong in all 4 runs (main + 3 seeds), 46 are right in all 4, and only 5 flip.
- The k = 5 → 20 F1 gain (about +0.02) is the same size as the seed-to-seed noise. With 75 test queries, F1 differences below about ±0.02 are treated as noise.

**Oracle upper bound (fit split only).** The oracle stops exactly when both gold passages are covered. It uses the labels, so no real method can beat it. On the 267 of 350 fit queries where both gold passages are in the top 20, it needs 591.5 tokens on average versus 2,558 for k = 20. **Maximum possible saving: 76.7%.**

## Known limitations

- `recall_at_budget` is a proxy: having the gold passages does not guarantee a correct answer. At k = 20, 15 of 75 test queries had both gold fed and still got F1 = 0.
- The calibration and test splits are small (75 queries each), so `ece` and `coverage` will be noisy.
- The conformal guarantee holds only when calibration and test queries come from the same distribution. Fit and calibration signal distributions look alike.
- 188 of 4,956 chunks are longer than the embedding model's 256-token limit and are truncated when embedded.
- 24% of fit queries never get both gold passages inside the top 20, so no budget can help them.
- `margin_mc` may point the wrong way on multi-hop data (see Key findings). The score model must be allowed to learn this.
- A 4B LLM is weak at multi-hop reasoning, so `answer_f1` is a secondary metric.
- The LLM answers some questions from memory (12 of 75 exactly right with no retrieval).
- Novelty versus existing work has not been checked yet.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## Reproduce (so far)

Run all commands from the repo root. Ollama must be running for the LLM steps.

```powershell
python src\data_loader.py               # 500-query HotpotQA subset + splits
python src\chunking.py                  # chunks + token_cost (Qwen3 tokenizer)
python src\index.py                     # embeddings + HNSW index
python notebooks\check_index.py         # index sanity + HNSW vs exact (fit only)
python src\baseline.py                  # fixed-k baseline, retrieval metrics (test)
python notebooks\oracle_fit.py          # oracle upper bound (fit only)
python tests\test_signals.py            # unit test for the signals
python src\signals.py                   # signals for all splits
python notebooks\plot_signals.py        # histograms (fit + calibration only)
python src\qa_metrics.py                # EM / F1 self-check
python notebooks\truncation_test.py     # truncation guard test
python notebooks\choose_num_ctx.py      # num_ctx from the largest prompt
python src\llm_baseline.py main         # LLM baseline, temperature 0 (test)
python src\llm_baseline.py qwen 1 2 3   # LLM baseline, Qwen's values, 3 seeds (test)
```
