# Index-Level Context Budgeting in Vector Search

**Goal:** minimize the context budget (in tokens) that a vector database feeds to an LLM in a RAG system, without losing the passage that holds the answer.

Standard RAG feeds a fixed number of passages (k) for every query. This project instead stores a `token_cost` for every chunk in the index and picks the most relevant chunks that fit inside a **token budget**. The budget is chosen per query from cheap retrieval signals and calibrated from data with conformal prediction.

> **Status: work in progress (draft README, updated 2026-11-08).** Done so far: data, chunks, index, fixed-k baseline (retrieval only), oracle upper bound, the confidence signals with a unit test, and a working local LLM call. Rows marked "not run yet" have no results.

## Progress

| Phase | What | Status |
|---|---|---|
| 0 | Setup (venv, requirements) | done |
| 1 | Data, chunks, `token_cost`, embeddings, HNSW index, index checks | done |
| 2 | Fixed-k baseline (retrieval metrics) | done |
| 2 | LLM for `answer_f1` | in progress: model chosen and a test call works; prompt template, `num_ctx` and truncation guard not done yet |
| 3 | Confidence signals | `signals.py` and unit test done; saving signals for all splits and histograms not done yet |
| 4-10 | Labels, score model, conformal budget, token-budget retrieval, the four checks, ablation | not started |

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
3. **Confidence score:** a logistic regression maps the signals to `conf_score`, the probability that the gold passage is covered. The rival to beat is a model using `top1_dist` only.
4. **Conformal budget:** on a held-out calibration split, pick a token budget `q_hat` per `conf_score` bin, so that the gold passage is inside the budget with probability about 1 - alpha.
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
| `token_cost` tokenizer | Qwen tokenizer (matches the answering LLM) |

HNSW was checked against exact search: recall@20 = 0.9994 on the fit split.

**Answering LLM.** Qwen3-4B-Instruct-2507, run locally with Ollama (`qwen3:4b-instruct-2507-q4_K_M`, model ID `0edcdef34593`, Q4_K_M, Ollama 0.40.15), through the native `/api/chat` endpoint (`src/llm.py`). It was chosen for speed and fit (8 GB VRAM). The project measures how much context to retrieve, not LLM strength.
- **Checked:** a test call returns a short answer with no `<think>` block, and `ollama ps` confirms that the `num_ctx` we send is applied. The model runs fully on the GPU (3.2 GB loaded).
- **Every option is sent explicitly in each call.** The model's built-in defaults are Qwen's values (temperature 0.7), so temperature 0 must be passed on purpose.
- `num_ctx` is the same for every method. The current 4096 is a test placeholder; the final value will be computed from the largest real prompt.
- *(not done yet)* A guard will compare Ollama's `prompt_eval_count` with our own token count to catch silent truncation. The chat template adds extra tokens (33 counted for a ~20-word prompt), so the guard flags only counts clearly **below** ours.
- *(not done yet)* The prompt template will be saved in the repo and be identical for every method.

**Two sampling settings.**

| Setting | Values | Used for |
|---|---|---|
| Main | temperature 0, seed 42, `num_predict` 256 | every headline number and the baseline vs method comparison |
| Secondary | Qwen's recommended values (temperature 0.7, top_p 0.8, top_k 20, min_p 0), several seeds (number to be decided) | error analysis and run-to-run variance only |

The two settings are never mixed in one table.

**Metrics.**
- **Primary:** `recall_at_budget` (share of queries where the gold passages are inside the fed set) and tokens fed.
- **Secondary:** `answer_f1` and exact match against the HotpotQA answer, after normalization. A 4B model is weak at multi-hop reasoning, so a low `answer_f1` may reflect the LLM, not retrieval.
- **No-retrieval baseline (k = 0):** measures how much the LLM answers from memory, since HotpotQA comes from Wikipedia.

**The four checks.**

| Check | Metric | Hoped-for result |
|---|---|---|
| 1. Does the score predict? | `auroc` | above 0.5 and above `top1_dist` alone |
| 2. Is it calibrated? | `ece` | close to 0 |
| 3. Is coverage right? | `coverage` | close to 1 - alpha |
| 4. Does it pay off? | `tokens_saved_pct` at equal `recall_at_budget` | positive versus fixed-k |

These are targets, not results.

**Rules.**
- Nothing is tuned on the test split.
- The oracle upper bound is computed on the fit split only.
- One thing changes between runs.
- Per-query logs are saved as CSV in `results/`, and every run is logged in [results/results.md](results/results.md).

## Results so far

**Fixed-k baseline (test split, 75 queries, retrieval only):**

| Method | recall (both gold) | recall (any gold) | avg tokens fed | answer_f1 |
|---|---|---|---|---|
| Fixed-k, k = 5 | 0.560 | 0.947 | 595.2 | not run yet |
| Fixed-k, k = 10 | 0.693 | 0.960 | 1274.1 | not run yet |
| Fixed-k, k = 20 | 0.773 | 0.973 | 2634.0 | not run yet |
| No retrieval, k = 0 | n/a | n/a | 0 | not run yet |
| Token-budget (conformal) | not run yet | not run yet | not run yet | not run yet |

**Oracle upper bound (fit split only).** The oracle stops exactly when both gold passages are covered. It uses the labels, so no real method can beat it. On the 267 of 350 fit queries where both gold passages are in the top 20, it needs 591.5 tokens on average versus k = 20. **This gives a maximum possible saving of 76.7%.**

Note: the token counts above were computed with the Qwen2.5 tokenizer and will be recomputed with the Qwen3 tokenizer.

## Known limitations

- `recall_at_budget` is a proxy: having the gold passage does not guarantee a correct answer.
- The calibration and test splits are small (75 queries each), so `ece` and `coverage` will be noisy.
- The conformal guarantee holds only when calibration and test queries come from the same distribution.
- 188 of 4,956 chunks are longer than the embedding model's 256-token limit and are truncated when embedded.
- HotpotQA needs two gold passages, which often sit close together at ranks 1 and 2. A tiny `margin_mc` may then mean "two good golds" rather than "unsure". This was seen in a few fit queries and will be checked on all of them.
- A 4B LLM is weak at multi-hop reasoning, so `answer_f1` is a secondary metric.
- Novelty versus existing work has not been checked yet.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## Reproduce (so far)

```powershell
python src\data_loader.py       # 500-query HotpotQA subset + splits
python src\chunking.py          # chunks + token_cost
python src\index.py             # embeddings + HNSW index
python notebooks\check_index.py # index sanity + HNSW vs exact (fit only)
python src\baseline.py          # fixed-k baseline (test)
python notebooks\oracle_fit.py  # oracle upper bound (fit only)
python tests\test_signals.py    # unit test for the signals
python src\signals.py           # signals demo on 3 fit queries
python src\llm.py               # one test call to the local LLM (Ollama must be running)
```

Run all commands from the repo root.
