# Run 22: Signal ablation (Phase 9.1)

Full write-up: [`ablations/PHASE9_REPORT.md`](../../ablations/PHASE9_REPORT.md). Data: `ablations/phase9_signals/summary.csv`.

## Goal
Remove each signal one at a time and see what changes: the score's AUROC, and the tokens used by the signal-based budgets (bins3, CQR).

## Setup
- Dev data only (fit + calibration, no test). Out-of-fold predictions; AUROC = mean over 5 CV seeds; budgets over 500 re-splits (75 / 350), alpha 0.1, method A.
- Sets: all 5, each "minus one", `top1_dist` only, all 5 + `q_comparison`, `q_comparison` only.
- `python ablations\phase9_signals\run_signal_ablation.py`

## What happened

| features | AUROC (OOF, fit) | bins3 tokens | CQR tokens |
|---|---|---|---|
| all 5 | 0.589 | 1,614 | 1,505 |
| − margin_mc | **0.480** | 1,614 | 1,485 |
| − var_local | 0.588 | 1,621 | 1,451 |
| − top1_dist | 0.592 | 1,627 | 1,478 |
| − elbow_idx | 0.596 | 1,639 | 1,508 |
| − entropy | 0.582 | 1,638 | 1,429 |
| top1_dist only | 0.461 | 1,655 | 1,491 |
| all 5 + q_comparison | **0.693** | 1,608 | 1,556 |
| q_comparison only | 0.677 | 1,605 | 1,432 |
| (global budget, no signals) | | 1,399 | 1,399 |

Coverage stays ~0.92 (bins3) and ~0.908 (CQR) in every row.

## Good findings
- **`margin_mc` carries almost all the information** of the distance signals: without it AUROC falls to 0.480 (worse than chance). The other four are interchangeable (±0.01), consistent with their ~0.9 correlation (run 08).
- **The question-type signal alone (0.677) is nearly as good as all six (0.693).**

## Surprising
- **For CQR, fewer features mean fewer tokens.** "q_comparison only" (1,432) and "− entropy" (1,429) come closest to the global budget. Extra weak features add noise to each query's budget, which pushes the average up.

## Stagnant / weak
- **No signal set beats the global budget** (1,399). bins3 is stuck at ~1,605-1,655 whatever the features: the per-bin small-sample penalty dominates.

## What went wrong
- Nothing.
