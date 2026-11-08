# Run notes

Detailed notes for every run: what was done, what happened, and what we learned.
`results/results.md` stays the short log of numbers. These files explain them.

Every file uses the same sections:
- **Goal**: why the run was done
- **Setup**: config and command
- **What happened**: the real output
- **Good findings**: results that help the project
- **Surprising**: results we did not expect
- **Stagnant / weak**: things that did not move, or signals that carry little information
- **What went wrong**: mistakes, bugs, rule breaks, and how they were fixed
- **Next**: what this run means for later tasks

Rule: only real output goes here. If something was not run, it says "not run yet".

## Index

| # | Date | Task | Run | Status |
|---|---|---|---|---|
| 01 | 2026-11-08 | 1.1-1.3 | [Data, splits and chunks](01_data_and_chunks.md) | done |
| 02 | 2026-11-08 | 1.4-1.7 | [Embeddings, HNSW index and index checks](02_index_and_checks.md) | done |
| 03 | 2026-11-08 | 2.1-2.2 | [Fixed-k baseline, retrieval only](03_fixed_k_retrieval.md) | done |
| 04 | 2026-11-08 | (extra) | [Oracle upper bound](04_oracle_upper_bound.md) | done (fit only) |
| 05 | 2026-11-08 | 1.3 / 2.3 | [Tokenizer switches](05_tokenizer_switches.md) | done |
| 06 | 2026-11-08 | 2.3 Step 3 | [First LLM call](06_llm_first_call.md) | done |
| 07 | 2026-11-08 | 3.1-3.2 | [Signals and unit test](07_signals_unit_test.md) | done |
| 08 | 2026-11-08 | 3.3-3.4 | [Signal distributions](08_signal_distributions.md) | done |
| 09 | 2026-11-08 | 2.3 Step 4 | [Truncation guard test](09_truncation_guard.md) | done |
| 10 | 2026-11-08 | 2.3 Step 5 | [Choosing num_ctx](10_num_ctx.md) | done |
| 11 | 2026-11-08 | 2.4 | [Fixed-k baseline with the LLM](11_llm_baseline_test.md) | done (main setting) |

## Biggest lessons so far

1. **Room to improve is large.** The oracle on fit shows the maximum possible saving is 76.7% of tokens versus k = 20 (run 04).
2. **`margin_mc` can point the wrong way on multi-hop data.** Perfect retrieval (both gold at ranks 1-2) has a smaller margin, not a larger one (runs 07, 08).
3. **Three signals are nearly the same signal.** `var_local`, `top1_dist` and `entropy` are correlated at about 0.9 (run 08).
4. **The second gold passage is the hard part.** At least one gold is found for ~95% of queries at k = 5, but both gold only for ~56% (run 03).
5. **Ollama did not silently truncate.** A too-long prompt gives an HTTP 400 error on Ollama 0.40.15 (run 09).
6. **More context barely improves answers.** k = 5 → 20 costs 4.4× the tokens for +0.028 F1 (run 11).
