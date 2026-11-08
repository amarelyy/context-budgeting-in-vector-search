# Run 19: Question-type signal (Phase 6, option 3)

Full report: [`ablations/phase6_methods/REPORT.md`](../../ablations/phase6_methods/REPORT.md).

## Goal
Add a stronger signal: is the question a "comparison" question (both entities named, so easy to retrieve) or a "bridge" question?

## Setup
- `src/question_type.py`: TF-IDF (word 1-2 grams) + logistic regression trained on fit questions to predict HotpotQA's "comparison" label. Output `q_comparison` = predicted probability; out-of-fold for fit, trained on all fit for the other splits. Saved to `data/processed/question_type.csv`.
- True type read from the cached HotpotQA download (`data/raw` unchanged).
- Then: score model, bins and CQR with 6 signals (`run_methods.py --qtype`, `cqr.py --qtype`). Best-case check with true types and no finite-sample penalty.

## What happened
- Classifier on fit (out-of-fold): AUROC 0.954, accuracy 0.857.
- Fit, by true type: bridge n 282, median tokens 437.5, p90 1,498; comparison n 68, median 181, p90 694. Spearman with tokens needed: −0.321.
- Score model AUROC (out-of-fold, fit): 0.609 → **0.704**.
- Dev estimate with 6 signals: 3 bins 1,608 tokens, CQR 1,556 tokens; global 1,399. All at ~0.91-0.92 coverage.
- Best case (true types, no penalty): per-type budget 1,320 vs global 1,356, **only 2.7% less**.

## Good findings
- The signal is real and strong (−0.32 vs ≤ 0.08 for the distance signals), and easy to detect from the question text.

## Surprising
- **A better score did not give a better budget.** At 90% coverage the budget is set by the hardest 10% of queries, which are bridge questions. The signal only makes the easy minority (comparison, ~20%) cheaper.
- CQR got slightly worse with the extra signal (1,505 → 1,556): budgets varied more without more accuracy in the tail.

## Stagnant / weak
- No per-query method beats the global budget at alpha 0.1, even with this signal. The best-case ceiling for this idea is 2.7%.

## What went wrong
- Nothing broke.
- Calibration has fewer comparison questions (13.3%) than fit (19.4%), a small shift from small samples.

## Next
- Princess decides: (A) report the global conformal budget + the explained negative result, (B) check lower coverage targets (alpha 0.3-0.5), or (C) more data. Must be decided before test.
