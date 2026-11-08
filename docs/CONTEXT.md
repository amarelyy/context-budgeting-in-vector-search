# CONTEXT.md: Full project details

Read this when you need the "why", the definitions, or the plan. Day-to-day work is tracked in `TASK.md`. Rules are in `CLAUDE.md`.

## 1. Summary
**Title:** Index-Level Context Budgeting in Vector Search
**Project code:** AI-B
**Owner:** rr (Ursula Maurentti Amarely), 2nd-year Information Engineering, DTETI, Universitas Gadjah Mada
**Purpose:** a project for rr's CV and research internship applications.
**Repo:** https://github.com/amarelyy/index-level-context-budgeting-in-vector-search

**One-sentence goal:** minimize the context budget (tokens) that a vector database feeds to an LLM in a RAG system, without losing the passage that holds the answer.

## 2. The problem
Standard RAG uses a fixed k: it always feeds the same number of passages, whatever the query.
1. **Budget waste.** If k is too large, the LLM reads noise. This costs tokens and latency, and may trigger the "lost in the middle" effect, where attention drops for content in the middle of a long context (Liu et al., 2024).
2. **Reliability blindness.** Vector similarity measures closeness, not reliability. A passage can be close in the vector space and still be wrong or uncertain. Standard retrieval has no signal for this.
3. **Count is the wrong unit.** "k passages" ignores that passages have different lengths. The real cost to an LLM is tokens.

## 3. Where this comes from (the UARP proposal)
The project grows out of rr's research proposal for the ANU Future Research Talent 2027 program: *Reliability-Aware Adaptive Retrieval (UARP)*. The proposal introduced two cheap signals computed from the distances of retrieved results:
- **Margin of confidence:** `margin_mc = d(q, v2) - d(q, v1)`. A large margin means a clear top match. Subtraction is used, not a ratio, because a ratio is unstable when `d(q, v1)` is near zero in normalized embedding spaces.
- **Local variance:** `var_local = (1/k) * sum((d(q, vi) - mean_d)^2)`. High variance means the neighbors are spread out, which may mean a noisy region.
- **Worked example (use as a unit test):** distances 0.20, 0.21, 0.90 give mean 0.4367, `margin_mc` = 0.01, `var_local` = 0.1074.
- **Gate from the proposal:** three Mc regions plus a variance override, with starting thresholds theta_high = 0.10, theta_low = 0.02, theta_var = 0.05. These are starting values, not final.

The goal has now shifted: instead of a retrieval loop that switches algorithms, the aim is to solve the budget problem deeper in the architecture.

## 4. What changed (the new direction)
| # | Change | Role |
|---|---|---|
| 1 | **Token-budget retrieval** | **Main change.** The index stores `token_cost` per chunk. Retrieval picks the most relevant chunks within a token budget, instead of a fixed count k. |
| 2 | **Conformal calibration** | Replaces hand-tuned thresholds. A cutoff is chosen from data so that the gold passage is inside the budget with probability about 1 - alpha. |
| 3 | **Index-time deduplication** | **Add-on.** Near-duplicate chunks are merged or removed when building the index, so the budget is not wasted on copies. |

**Out of scope for now:** in-index early termination of the HNSW walk (stretch idea), multi-resolution storage (sentence/paragraph/section), UNG algorithm switching, Self-RAG/FLARE/Xiong-Zhang baselines, C++ integration. Novelty versus existing work has **not been checked yet**. Do not claim novelty in the README until a literature check is done.

## 5. The five base components
Each listing may swap at most one. Here, component 2 is swapped.

| # | Component | In this project |
|---|---|---|
| 1 | Small RAG pipeline with FAISS HNSW | As planned |
| 2 | Adaptive-k gate (margin and variance) | **Swapped:** token-budget gate with conformal calibration |
| 3 | Compare recall, answer accuracy, token count against fixed-k on 2 QA datasets | As planned (HotpotQA first, a BEIR subset second) |
| 4 | Failure analysis, including "lost in the middle" | As planned |
| 5 | Ablation on each gate threshold | Adapted: ablate each signal and sweep alpha, and compare conformal with grid search |

## 6. The confidence score: where it comes from
**Signals** (free at retrieval time, from the retrieved distances): `margin_mc`, `var_local`, `top1_dist`, `elbow_idx`, `entropy`.
**Label** (what "correct" means): whether the gold passage is inside the fed set. HotpotQA provides gold supporting facts; BEIR provides qrels. This is a **proxy**: having the gold passage does not guarantee the LLM answers correctly. Report both.

**Pipeline:**
1. For each query, compute the signals.
2. Compute `gold_pos_tokens` from the labels.
3. Fit a small model (logistic regression) on the **fit** split. Input: signals. Target: gold covered within a given position. Output: `conf_score` between 0 and 1. A second model using only `top1_dist` is the rival to beat.
4. On the **calibration** split, take the (1 - alpha) quantile of `gold_pos_tokens` with the finite-sample correction, giving `q_hat`. Use separate `q_hat` values for bins of `conf_score` (Mondrian conformal). alpha = 0.1 is a starting value.
5. At test time: signals -> `conf_score` -> bin -> `q_hat` -> select chunks until the token budget is full.

**Caveat:** the conformal guarantee holds on average, and only if calibration and test queries come from the same distribution. It may fail on a different dataset. That is exactly what the transfer test checks.

## 7. The four checks (how we know the score works)
Run in order. If check 1 fails, stop and discuss before continuing.
| Check | Output | Good result |
|---|---|---|
| 1. Does the score predict? | `auroc` | clearly above 0.5, and above `top1_dist` alone |
| 2. Is it calibrated? | `ece` (and a reliability diagram) | close to 0 |
| 3. Is coverage right? | `coverage` | close to 1 - alpha |
| 4. Does it pay off? | `tokens_saved_pct` at equal `recall_at_budget` | positive versus fixed-k |

All "good result" targets are what we hope to see. They are not results.

## 8. Data, models, and setup
- **Primary dataset:** HotpotQA (multi-hop QA with gold supporting facts). Start with a small subset (a few hundred queries is a starting value).
- **Second dataset:** a BEIR subset (to be chosen; confirm format and qrels when loading).
- **Index:** FAISS HNSW (`faiss-cpu`). Embeddings are L2-normalized. Record which distance is used (L2 or inner product) because `margin_mc` depends on it.
- **Baseline:** fixed-k with k = 5, 10, 20, reported first.
- **Splits:** fit / calibration / test with a fixed seed. Query IDs saved. The test split is not touched until evaluation.
- **Open choices (decide and record in the Decisions Log in `TASK.md`):** embedding model, tokenizer for `token_cost`, whether and which LLM is used for `answer_f1`, HNSW parameters (M, efConstruction, efSearch), the number of candidates retrieved per query for the signals, how multi-hop gold passages are handled in `gold_pos_tokens`.

## 9. Known risks and open questions
1. Signals computed on only a few points may be too noisy to trust. Check how stable they are.
2. A threshold or `q_hat` calibrated on one dataset may not transfer to another.
3. "Gold passage inside the set" is a proxy for a correct answer.
4. Small calibration sets make `ece` and coverage noisy. Report how many queries were used.
5. HNSW is approximate. Recall of the index itself should be checked against exact search on a small sample.
6. Different tokenizers give different `token_cost`. Fix one and keep it.
7. Novelty is unchecked (see Section 4).

## 10. Success criteria
- A working pipeline with a fixed-k baseline and the budgeted method on the same test queries.
- Real, logged numbers for the four checks.
- A README with a results table and plots that anyone can reproduce.
- Every CV line is backed by a number rr can explain, including why it changed.
