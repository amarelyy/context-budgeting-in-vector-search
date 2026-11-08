# TASK.md: Project workflow and progress

**READ THIS FIRST before starting any work.** Then read `CLAUDE.md` (rules) and `CONTEXT.md` (details).
**After any work:** update the status, the Progress Log, and (if needed) the Troubleshooting and Decisions logs.

## Status legend
- `[ ]` NOT YET: not started
- `[~]` IN PROGRESS: started, not finished (only one or two tasks at a time)
- `[x]` DONE: code ran, rr saw real output, result logged, committed
- `[!]` TROUBLED: blocked. See the Troubleshooting Log.

Do tasks **in order**. Do not skip a phase unless rr says so. A task marked with **GATE** must pass before moving on.

---

## Phase 0: Setup
- [x] **0.1** Clone the repo to `C:\Users\ASUS TUF\index-level-context-budgeting-in-vector-search`
- [x] **0.2** Create the folder structure and `.gitignore`
- [x] **0.3** Remind user to push the structure to GitHub (rr reported success; verify with `git log`)
- [x] **0.4** Install Claude Code, run `claude` in the repo, add `CLAUDE.md`, `CONTEXT.md`, `TASK.md`
- [x] **0.5** Create `.venv`, install packages, write `requirements.txt` (proposed: faiss-cpu, numpy, pandas, scikit-learn, datasets, sentence-transformers, matplotlib; confirm each before adding)
- [x] **0.6** Write the README skeleton: title, one-paragraph goal, empty results table
- [x] **0.7** Create `results/results.md` with a header and the log format (date, config, numbers, why it changed)

## Phase 1: Data and index
- [x] **1.1** `data_loader.py`: load a small HotpotQA subset with gold supporting facts (start with a few hundred queries, a starting value). Print one example to check.
- [x] **1.2** Make the fit / calibration / test splits with a fixed seed. Save query IDs to `data/processed/`. (Split ratio is a starting value to decide.)
- [x] **1.3** `chunking.py`: build chunks from the passages and compute `token_cost`. Record the tokenizer in the Decisions Log.
- [x] **1.4** Embed the chunks and L2-normalize. Record the embedding model in the Decisions Log. Save to `data/processed/`.
- [x] **1.5** `index.py`: build the FAISS HNSW index and write a `search(query, top_m)` that returns chunk IDs and distances. Record the HNSW parameters.
- [x] **1.6** Sanity check: run a few queries by hand, confirm results look sensible. Record the distance type (L2 or inner product).
- [x] **1.7** Check index recall against exact search on a small sample (see Risk 5 in `CONTEXT.md`).

## Phase 2: Fixed-k baseline (report this first)
- [x] **2.1** `baseline.py`: retrieve with k = 5, 10, 20 for the test queries.
- [x] **2.2** Compute `recall_at_k` (gold passages covered) and average `total_tokens` for each k.
- [~] **2.3** Decide whether to use an LLM for `answer_f1`. If yes, record which one and the prompt template. If no, note that only retrieval metrics are reported.
- [ ] **2.4** Log the baseline table in `results/results.md`. **GATE:** the baseline must exist before Phase 5.

## Phase 3: Signals
- [x] **3.1** `signals.py`: compute `margin_mc`, `var_local`, `top1_dist`, `elbow_idx`, `entropy` per query. Record how many candidates (top_m) are used.
- [x] **3.2** Unit test with the worked example: distances 0.20, 0.21, 0.90 must give `margin_mc` = 0.01 and `var_local` ≈ 0.1074.
- [ ] **3.3** Save signals for all three splits.
- [ ] **3.4** Plot the signal distributions (histograms) and check for anything strange.

## Phase 4: Labels
- [ ] **4.1** `labels.py`: compute `gold_pos_tokens`. Decide how to handle multi-hop questions (two gold passages) and record it.
- [ ] **4.2** Decide how to handle queries where gold is not found in the top_m. Record it.
- [ ] **4.3** Save labels for all splits.

## Phase 5: Score model
- [ ] **5.1** `score_model.py`: fit logistic regression on the **fit** split. Target and threshold position are decided and recorded.
- [ ] **5.2** Fit the rival model using `top1_dist` only.
- [ ] **5.3** Produce `conf_score` for all splits and save.

## Phase 6: Conformal budget
- [ ] **6.1** `conformal.py`: compute `q_hat` on the **calibration** split with the finite-sample correction (alpha = 0.1, a starting value).
- [ ] **6.2** Make bins of `conf_score` (number of bins is a starting value) and one `q_hat` per bin.
- [ ] **6.3** Sanity check: coverage on the calibration split itself.

## Phase 7: Token-budget retrieval
- [ ] **7.1** `budget_retrieval.py`: for each query, get `conf_score` -> bin -> `q_hat`, then add chunks in ranked order until the token budget is full. (Starting approach. A value-per-cost variant can be tried later.)
- [ ] **7.2** Run on the **test** split for the first time. Log the config and numbers.
- [ ] **7.3** Compare against the fixed-k baseline on the same queries. Log the table.

## Phase 8: The four checks
- [ ] **8.1** `auroc` of `conf_score` against the gold-covered label, and against the `top1_dist` rival. **GATE:** if it does not beat the rival, stop and discuss with rr.
- [ ] **8.2** `ece` and a reliability diagram.
- [ ] **8.3** `coverage` on the test split versus 1 - alpha.
- [ ] **8.4** `tokens_saved_pct` at equal `recall_at_budget` versus fixed-k (and `answer_f1` if an LLM is used).

## Phase 9: Ablation, failure analysis, add-on
- [ ] **9.1** Ablation: remove each signal one at a time and record what changes.
- [ ] **9.2** Sweep alpha and plot sensitivity curves. Compare conformal with a simple grid-search threshold.
- [ ] **9.3** Failure analysis: look at queries where gold was missed. Include the "lost in the middle" check (where gold sits in the context).
- [ ] **9.4** `dedup.py`: add index-time deduplication, rebuild, re-run Phase 7-8, and compare with and without.
- [ ] **9.5** (Stretch) Transfer test on a BEIR subset: do `q_hat` and the score model work without re-calibration?

## Phase 10: Wrap-up
- [ ] **10.1** Final README: method, results tables, plots, how to reproduce.
- [ ] **10.2** Write CV bullet(s). Every number must be backed by `results/results.md`, and rr must be able to explain why it changed.
- [ ] **10.3** Clean up: `requirements.txt` correct, no data or secrets committed, final push.
- [ ] **10.4** Literature check for novelty before claiming anything as new.

---

## Progress log (newest first)
| Date | Task | What happened |
|---|---|---|
| 2026-11-08 | 2.3 Step 3 | `src/llm.py`: `chat(messages, options)` calls Ollama native `/api/chat` with all options explicit (main: temperature 0, seed 42, num_ctx 4096 placeholder, num_predict 256). Test "capital of France" returned "Paris", done_reason "stop", no `<think>` block. prompt_eval_count = 33 (includes chat template tokens). `ollama ps`: CONTEXT 4096 (num_ctx applied), 100% GPU, 3.2 GB loaded. First call load time 4.2 s, answer itself < 0.1 s. Not committed. |
| 2026-11-08 | 3.1-3.2 | `src/signals.py` computes margin_mc, var_local, top1_dist, elbow_idx, entropy from the top_m = 20 distances. `tests/test_signals.py` (plain asserts, no pytest): worked example 0.20/0.21/0.90 gives margin_mc 0.01, var_local 0.1074, mean 0.4367, elbow_idx 2. All 3 tests pass. Demo on 3 fit queries looks sensible. Observation: in multi-hop queries the 2 golds can sit at ranks 1-2 with a tiny margin_mc (Saab 36 query: 0.007), so a small margin may mean "two good golds", not "unsure". Check in 3.4 / 8.1. Not committed: Princess will commit. |
| 2026-11-08 | 2.3 (partial), 2.2 rerun | rr chose an LLM (local Qwen2.5 7B via Ollama). Switched `token_cost` to the Qwen2.5 tokenizer and reran chunking and baseline: same recall, tokens ~10% higher (k=20 mean 2634). Oracle moved to fit only (`notebooks/oracle_fit.py`): max saving 76.7% vs k=20. An earlier oracle run on test was deleted from the log. 2.3 still open: Ollama model tag and prompt template. Not committed: rr will commit. |
| 2026-11-08 | 2.1-2.2 | Fixed-k baseline on test (75 queries). Both-gold recall k=5/10/20 = 0.560/0.693/0.773, any-gold = 0.947/0.960/0.973, mean total_tokens = 539/1153/2379. Logged in `results/results.md`. Note: CLAUDE.md says "don't touch test before Phase 7" but TASK 2.1 says baseline on test; followed TASK (fixed-k has nothing to tune). Not committed yet: rr will commit. |
| 2026-11-08 | 1.4-1.7 | Embedded 4956 chunks with all-MiniLM-L6-v2 (dim 384, all norms 1.0), saved `chunk_embeddings.npy`. Built HNSW (M 32, efC 200, efS 100), saved `hnsw.index`. Sanity check on 3 fit queries: both gold at ranks 1-2 each time. Fit split: both gold in top 5/10/20 = 0.520/0.683/0.763. HNSW vs exact recall@20 on 350 fit queries = 0.9994 (min 0.95), top-1 identical 100%. 188 chunks are longer than the model's 256-token limit and are truncated. Checks in `notebooks/check_index.py` (fit queries only). |
| 2026-11-08 | 1.1-1.3 | Downloaded HotpotQA (distractor, validation) via `datasets`. Sampled 500 queries (seed 42) to `data/raw/hotpot_subset.jsonl`. Splits 350/75/75 saved to `data/processed/splits.json`. Built 4956 chunks (one paragraph = one chunk, deduplicated by title+text) with `token_cost` (median 107, mean 119.7, max 1846; 593,478 tokens total). Every query has exactly 2 gold chunks. Saved `chunks.jsonl` and `gold_chunk_ids.json`. |
| 2026-11-08 | 0.5-0.7 | Created `.venv` (Python 3.11.9), installed the 7 proposed packages (all import OK), pinned versions in `requirements.txt`. Wrote README skeleton and `results/results.md`. Phase 0 done pending push. |
| 2026-11-08 | 0.1-0.3 | Repo cloned, folder structure created and pushed. Project files written. |

## Troubleshooting log
| Date | Task | Error / problem | What was tried | Status |
|---|---|---|---|---|
| | | | | |

## Decisions log
| Date | Decision | Value chosen | Why |
|---|---|---|---|
| 2026-11-08 | Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, dim 384, L2-normalized (starting value) | Small, fast on CPU, same tokenizer as `token_cost`. Max 256 tokens: 188 of 4956 chunks are truncated. |
| 2026-11-08 | Distance type | L2 distance on unit vectors = sqrt of FAISS squared L2, range [0, 2], smaller = closer | `margin_mc` and `var_local` use these distances. Note L2^2 = 2 - 2*cos. |
| 2026-11-08 | Tokenizer for `token_cost` | ~~all-MiniLM-L6-v2~~ → `Qwen/Qwen2.5-7B-Instruct` tokenizer | Changed after decision 2.3 (use an LLM): count tokens the way the answering LLM does. All Qwen2.5 sizes share this tokenizer. Chunks unchanged, so the embeddings and index stay valid. Corpus total 593,478 → 651,022 tokens. |
| 2026-11-08 | Oracle upper bound | Computed on **fit only**, never on test | rr's rule. The oracle is the maximum possible result (76.7% max saving vs k=20 on fit). |
| 2026-11-08 | Subset size | 500 queries, seed 42 (starting value) | Small enough to run fast on CPU. |
| 2026-11-08 | Chunk definition | 1 paragraph = 1 chunk, one shared corpus for all queries | Simplest. HotpotQA gold is given per paragraph. Long paragraphs (max 1846 tokens) will be truncated by the embedding model in 1.4. |
| 2026-11-08 | LLM for `answer_f1` | **Qwen3-4B-Instruct-2507**, run locally with Ollama (native `/api/chat`, not `/v1`). Ollama tag `qwen3:4b-instruct-2507-q4_K_M`, ID `0edcdef34593`, 2.5 GB, Q4_K_M, 4.0B params, model context length 262144, Ollama 0.40.15 (checked 2026-11-08 with `ollama list` / `ollama show`). Modelfile defaults = Qwen's values (temperature 0.7, top_k 20, top_p 0.8, repeat_penalty 1), so temperature 0 must be passed explicitly. `ollama show` lists a "thinking" capability: check in Step 3 that no `<think>` block appears. `ollama ps`: model not loaded yet, CONTEXT to be confirmed in Step 3. num_ctx: set explicitly (not the default); value computed from real data later, starting value. Sampling: **two settings** (see the next row). num_predict: small (e.g. 256, starting value). Prompt template: TBD, saved in repo, same for every method. | Official Instruct version, non-thinking only (no `<think>` blocks), so answers are short and stable. 4B fits in 8 GB VRAM and runs fast for many reruns; the project is about retrieval budget, not LLM strength. Report wording: "chosen for speed and fit" (do NOT claim it beats 8B). Replaces the earlier Qwen2.5 7B plan. `recall_at_budget` = primary metric, `answer_f1` = secondary. |
| 2026-11-08 | HNSW parameters (M, efConstruction, efSearch) | 32 / 200 / 100 (starting values) | Common defaults. Recall vs exact = 0.9994 on fit, so no tuning needed for now. |
| 2026-11-08 | LLM sampling settings | **Main:** temperature 0, fixed seed (seed value TBD, starting value): used for every headline number and the baseline vs method comparison. **Secondary:** Qwen's values (temperature 0.7, top_p 0.8, top_k 20, min_p 0), several seeds (count TBD, starting value): used only for error analysis and run-to-run variance. Every setting passed explicitly in each call. Per-query logs saved as CSV in `results/`. | Princess's choice. Temperature 0 makes differences come from retrieval, not luck. Qwen's values show how stable `answer_f1` is and let us compare errors between the two. Never mix the two settings in one table. |
| 2026-11-08 | top_m candidates per query | 20 (starting value) | Same pool as k=20 and the oracle. `var_local` and `entropy` use all 20; `margin_mc` uses ranks 1-2. |
| 2026-11-08 | Signal definitions | `var_local` = population variance (1/m) of the top_m distances. `elbow_idx` = number of chunks before the biggest jump between neighbouring distances (1-based). `entropy` = entropy of softmax(-d / tau) divided by log(m), so it is in [0, 1]; tau = 0.1 (starting value) | tau = 1 would make the softmax almost flat (distances only span ~0.4-1.1), so entropy would be ~1 for every query. Check the histogram in 3.4. |
| 2026-11-08 | Split ratio (fit / calibration / test) | 70 / 15 / 15 = 350 / 75 / 75 queries (starting value) | Calibration and test are small, so `ece` and `coverage` will be noisy (Risk 4). Report the counts. |
| | alpha, number of bins | TBD (starting: alpha 0.1) | |
| | Multi-hop gold handling | TBD (baseline reports both "both gold" and "any gold" for now) | |

## Reminders for Princess (remind every session until done)
- [~] **Update README for the testing method** (draft written 2026-11-08, not committed; finalize after the LLM runs and the Qwen3 tokenizer switch): LLM (Qwen3-4B-Instruct-2507 via Ollama, "chosen for speed and fit"), the two sampling settings (temperature 0 main, Qwen's values for error analysis), `recall_at_budget` primary / `answer_f1` secondary, the per-query CSV logs in `results/`, the oracle upper bound (fit only). Do it after the LLM pipeline runs (after task 2.3 code).
- [ ] Commit and push the pending changes.

## Open questions for rr
- Which BEIR subset to use as the second dataset?
- Is an LLM available for answer quality, or is the project retrieval-only at first?
