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
| 12 | 2026-11-08 | 2.4 | [LLM baseline, Qwen's values, 3 seeds](12_llm_baseline_qwen_seeds.md) | done (secondary setting) |
| 13 | 2026-11-08 | 4.1-4.3 | [Labels](13_labels.md) | done |
| 14 | 2026-11-08 | 6 (ablation) | [Phase 6 coverage ablation](14_phase6_coverage_ablation.md) | done (fit + calibration only) |
| 15 | 2026-11-08 | (preview) | [Signal preview: do signals predict tokens needed?](15_signal_preview_fit.md) | done (fit only) |
| 16 | 2026-11-08 | 5.1-5.3 | [Score model](16_score_model.md) | done (judged on fit only) |
| 17 | 2026-11-08 | 6.1-6.3 | [Conformal budget per bin + dev estimate](17_conformal_mondrian.md) | done; **negative result** |
| 18 | 2026-11-08 | 6 (option 2) | [CQR](18_cqr.md) | done; no gain over global |
| 19 | 2026-11-08 | 6 (option 3) | [Question-type signal](19_question_type_signal.md) | done; strong signal, no budget gain |
| 20 | 2026-11-08 | 7.1-7.3 | [Token-budget retrieval on test](20_budget_retrieval_test.md) | done; coverage holds, no gain over calibrated fixed-k |
| 21 | 2026-11-08 | 8.1-8.4 | [The four checks on test](21_four_checks_test.md) | done; 1 weak pass, 2 trivial pass, 3 pass, 4 fail |

## Biggest lessons so far

1. **Room to improve is large.** The oracle on fit shows the maximum possible saving is 76.7% of tokens versus k = 20 (run 04).
2. **`margin_mc` can point the wrong way on multi-hop data.** Perfect retrieval (both gold at ranks 1-2) has a smaller margin, not a larger one (runs 07, 08).
3. **Three signals are nearly the same signal.** `var_local`, `top1_dist` and `entropy` are correlated at about 0.9 (run 08).
4. **The second gold passage is the hard part.** At least one gold is found for ~95% of queries at k = 5, but both gold only for ~56% (run 03).
5. **Ollama did not silently truncate.** A too-long prompt gives an HTTP 400 error on Ollama 0.40.15 (run 09).
6. **More context barely improves answers.** k = 5 → 20 costs 4.4× the tokens for +0.028 F1 (run 11), and that gain is about the size of seed-to-seed noise (±0.02, run 12).
7. **LLM errors are stable.** Only 2-5 of 75 queries flip between right and wrong across 4 runs (run 12).
8. **24% of queries can never be covered** within the top 20, so 90% coverage (alpha = 0.1) is impossible unconditionally. Phase 6 must handle this (run 13).
9. **A single global token budget is no better than fixed-k** at equal coverage. Any efficiency gain must come from the per-query signals (run 14).
10. **Warning: the signals barely predict how many tokens a query needs** (|Spearman| ≤ 0.08 on fit; best single-signal AUROC 0.58 for "easy" queries). Check 1 (8.1) may fail (run 15).
11. **The combined score is weak but real.** CV AUROC 0.589 on fit, above the `top1_dist` rival (0.460). The model learned a negative weight on `margin_mc` by itself (run 16).
12. **Per-bin budgets cost tokens instead of saving them** (global 1,399 → 3 bins 1,614 tokens on dev data). Each small bin pays its own finite-sample penalty, and the weak score cannot make up for it. The planned method does not beat fixed-k yet (run 17).
13. **CQR fixes the bin problem but still loses to the global budget** (1,505 vs 1,399). A weak predictor makes per-query budgets worse than one shared budget (run 18).
15. **On test, the conformal guarantee holds (0.931 vs 0.90) and the global budget saves 41% vs k = 20, but a fixed-k chosen on calibration by the same conformal rule (k = 12) does as well with more coverage.** The useful part is the conformal choice of size, not counting in tokens (run 20).
16. **The four checks on test:** score AUROC 0.677 (weak pass, not significantly above `top1_dist`); ECE 0.038 (pass, but the score barely spreads); coverage 0.931 vs 0.90 (pass); `tokens_saved_pct` vs fixed-k at equal recall −9.8% (fail) (run 21).
14. **At 90% coverage, the hardest 10% of queries set the budget, and they are all bridge questions.** A strong question-type signal (score AUROC 0.609 → 0.704) still gives no gain; even a perfect per-type budget would save only 2.7% (run 19).
