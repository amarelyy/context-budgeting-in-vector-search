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
- [ ] **1.1** `data_loader.py`: load a small HotpotQA subset with gold supporting facts (start with a few hundred queries, a starting value). Print one example to check.
- [ ] **1.2** Make the fit / calibration / test splits with a fixed seed. Save query IDs to `data/processed/`. (Split ratio is a starting value to decide.)
- [ ] **1.3** `chunking.py`: build chunks from the passages and compute `token_cost`. Record the tokenizer in the Decisions Log.
- [ ] **1.4** Embed the chunks and L2-normalize. Record the embedding model in the Decisions Log. Save to `data/processed/`.
- [ ] **1.5** `index.py`: build the FAISS HNSW index and write a `search(query, top_m)` that returns chunk IDs and distances. Record the HNSW parameters.
- [ ] **1.6** Sanity check: run a few queries by hand, confirm results look sensible. Record the distance type (L2 or inner product).
- [ ] **1.7** Check index recall against exact search on a small sample (see Risk 5 in `CONTEXT.md`).

## Phase 2: Fixed-k baseline (report this first)
- [ ] **2.1** `baseline.py`: retrieve with k = 5, 10, 20 for the test queries.
- [ ] **2.2** Compute `recall_at_k` (gold passages covered) and average `total_tokens` for each k.
- [ ] **2.3** Decide whether to use an LLM for `answer_f1`. If yes, record which one and the prompt template. If no, note that only retrieval metrics are reported.
- [ ] **2.4** Log the baseline table in `results/results.md`. **GATE:** the baseline must exist before Phase 5.

## Phase 3: Signals
- [ ] **3.1** `signals.py`: compute `margin_mc`, `var_local`, `top1_dist`, `elbow_idx`, `entropy` per query. Record how many candidates (top_m) are used.
- [ ] **3.2** Unit test with the worked example: distances 0.20, 0.21, 0.90 must give `margin_mc` = 0.01 and `var_local` ≈ 0.1074.
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
| 2026-11-08 | 0.5-0.7 | Created `.venv` (Python 3.11.9), installed the 7 proposed packages (all import OK), pinned versions in `requirements.txt`. Wrote README skeleton and `results/results.md`. Phase 0 done pending push. |
| 2026-11-08 | 0.1-0.3 | Repo cloned, folder structure created and pushed. Project files written. |

## Troubleshooting log
| Date | Task | Error / problem | What was tried | Status |
|---|---|---|---|---|
| | | | | |

## Decisions log
| Date | Decision | Value chosen | Why |
|---|---|---|---|
| | Embedding model | TBD | |
| | Tokenizer for `token_cost` | TBD | |
| | LLM for `answer_f1` (or none) | TBD | |
| | HNSW parameters (M, efConstruction, efSearch) | TBD | |
| | top_m candidates per query | TBD | |
| | Split ratio (fit / calibration / test) | TBD | |
| | alpha, number of bins | TBD (starting: alpha 0.1) | |
| | Multi-hop gold handling | TBD | |

## Open questions for rr
- Which BEIR subset to use as the second dataset?
- Is an LLM available for answer quality, or is the project retrieval-only at first?
