# Index-Level Context Budgeting in Vector Search

**Goal:** minimize the context budget (in tokens) that a vector database feeds to an LLM in a RAG system, without losing the passage that holds the answer.

Standard RAG feeds a fixed number of passages (k) for every query. This project instead stores a `token_cost` for every chunk in the index and picks the most relevant chunks that fit inside a **token budget**. The budget is chosen per query from cheap retrieval signals and calibrated from data with conformal prediction.

> **Status: work in progress (draft README).** Only the fixed-k baseline and the oracle upper bound have been run. Rows marked "not run yet" have no results.

## Method (planned)

1. **Index:** FAISS HNSW over L2-normalized embeddings. Each chunk stores its `token_cost`.
2. **Signals** (free at retrieval time, from the retrieved distances): `margin_mc`, `var_local`, `top1_dist`, `elbow_idx`, `entropy`.
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

**Answering LLM.** Qwen3-4B-Instruct-2507, run locally with Ollama (`qwen3:4b-instruct-2507-q4_K_M`, Q4_K_M, Ollama 0.40.15), through the native `/api/chat` endpoint. It was chosen for speed and fit (8 GB VRAM). The project measures how much context to retrieve, not LLM strength.
- `num_ctx` is set explicitly and is the same for every method. Its value is computed from the largest real prompt.
- A guard compares Ollama's `prompt_eval_count` with our own token count, to catch silent truncation.
- The prompt template is saved in the repo and is identical for every method.

**Two sampling settings.**

| Setting | Values | Used for |
|---|---|---|
| Main | temperature 0, fixed seed | every headline number and the baseline vs method comparison |
| Secondary | Qwen's recommended values (temperature 0.7, top_p 0.8, top_k 20, min_p 0), several seeds | error analysis and run-to-run variance only |

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
```
