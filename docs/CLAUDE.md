# CLAUDE.md: Rules for Claude Code

Project: **Index-Level Context Budgeting in Vector Search** (AI-B)
Owner: rr, 3nd-year Information Engineering student Built for CV and research internship applications.

## 0. Before you do ANYTHING (every session)
1. Read `TASK.md` first. Find the current task and its status.
2. Read `CONTEXT.md` if you need the "why" or the definitions.
3. Tell rr in 1-2 lines which task you are about to work on. Then start.
4. When you stop, update `TASK.md` (status, progress log, problems). Never end a session without this.
5. Never push a work on your own, just tell user and remind to push. Reminder on every section or on every around 3 subsection, so tracked easily.

## 1. How to talk to rr
- Use **simple English**, short sentences, step by step. rr may mix in Indonesian; reply in simple English unless asked otherwise.
- Before any code, say in 1-2 lines **what it does and why**.
- Give **small snippets** that rr can run and test one at a time. Do **not** rewrite whole files unless asked.
- Keep answers short and genuinely understood. No over-engineering. If a simple way works, use it.
- rr must be able to explain every number and every line. If something is complicated, explain it first.

## 2. Honesty rules (most important)
- **Never invent results.** If you did not run the code, say "not run yet" and ask rr to run it and paste the output.
- Any number you guess (thresholds, alpha, k, bin count, split ratio, M, efSearch) is a **starting value to be tuned**. Label it that way.
- If a dataset, library, or API is unclear, **check it** and say what you confirmed and what you did not.
- If a result looks too good, or a check fails, say so. Do not hide it or tune until it passes.
- Do not claim something works until the matching check in `TASK.md` has real output.

## 3. Experiment rules
- Three splits: **fit / calibration / test**. Fixed random seed. Save the query IDs. **Never tune anything on the test split.** The test split is used for the fixed-k baseline (Phase 2, nothing to tune) and from Phase 7 onward. Thresholds, alpha, bins and models are chosen on fit/calibration only. The **oracle** (upper bound) is computed on fit only, never on test.
- Never change rr's Ollama models (no pull/rm/create). rr manages them. Read-only checks (`ollama list`) are fine.
- Report the **fixed-k baseline first**, then the adaptive method, on the **same queries**.
- Only **one** of the five base components may be swapped (the gate, component 2). Do not add other swaps.
- Log every result in `results/results.md`: date, config (k, alpha, bins, seed, split), the numbers, and a one-line "why did it change".
- For every run, also write a detailed note in `results/runs/NN_name.md` (goal, setup, what happened, good findings, surprising, stagnant / weak, what went wrong, next) and add it to the index in `results/runs/README.md`.
- Change **one thing at a time** between runs, so rr can explain the difference.
- Use the exact variable names in the glossary (Section 6). Do not rename them.

## 4. Scope and safety
- Stay inside the plan in `CONTEXT.md`. Out of scope: Self-RAG, FLARE, Xiong-Zhang router, UNG switching, C++ modules.
- Ask before adding a new dependency. Keep `requirements.txt` up to date when one is added.
- Never edit or delete `data/raw/`. Never commit data files or large files (see `.gitignore`).
- No secrets in code. API keys go in environment variables, never in the repo.
- Git: commit when a task is DONE, with a short message such as `Phase 2: fixed-k baseline`. Ask rr before `git push`. Never force-push.

## 5. Environment
- rr uses **Windows with PowerShell** and VS Code. Use PowerShell syntax in terminal commands (not `touch` or `mkdir -p`).
- Python in a virtual environment (`.venv`, not committed). Start with CPU only (`faiss-cpu`).
- Repo: https://github.com/amarelyy/index-level-context-budgeting-in-vector-search

## 6. Folder map and variable glossary
```
src/
  data_loader.py       Phase 1: load HotpotQA subset, make splits
  chunking.py          Phase 1: make chunks, token_cost
  index.py             Phase 1: FAISS HNSW build + search
  baseline.py          Phase 2: fixed-k (5, 10, 20)
  signals.py           Phase 3: confidence signals
  labels.py            Phase 4: gold_pos_tokens
  score_model.py       Phase 5: conf_score
  conformal.py         Phase 6: q_hat
  budget_retrieval.py  Phase 7: token-budget retrieval
  dedup.py             Phase 9: index-time dedup (add-on)
  evaluate.py          Phase 8: the four checks
data/raw, data/processed   (not pushed)
notebooks/                 quick experiments
results/                   results.md, tables, plots
```

| Variable | Meaning |
|---|---|
| `token_cost` | number of tokens in a chunk |
| `margin_mc` | distance gap between top-1 and top-2 |
| `var_local` | variance of the top retrieved distances |
| `top1_dist` | top-1 distance alone (rival signal to beat) |
| `elbow_idx` | where the sorted scores drop sharply |
| `entropy` | softmax entropy over the scores |
| `conf_score` | predicted probability that the gold passage is covered |
| `gold_pos_tokens` | token position in the ranked list where the gold passage is covered |
| `q_hat` | conformal token budget |
| `alpha` | allowed miss rate (starting value 0.1) |
| `recall_at_budget` | share of queries where gold is inside the fed set |
| `auroc`, `ece`, `coverage`, `tokens_saved_pct` | outputs of the four checks |
| `answer_f1` | answer quality (if an LLM is used) |

## 7. When something goes wrong
- Try at most **2 fixes**. If it still fails, stop.
- Set the task to **TROUBLED** in `TASK.md`, write the error and what you tried in the Troubleshooting Log, and ask rr.
- Do not work around a problem by changing the plan, the splits, or the metric. Ask first.

## 8. Definition of done (for any task)
A task is DONE only when: the code runs, rr has seen the real output, the result is logged in `results/results.md` (if it produces a number), `TASK.md` is updated, and the change is committed.
