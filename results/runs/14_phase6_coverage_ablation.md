# Run 14: Phase 6 ablation, coverage handling and alpha

Full report, tables and plot: [`ablations/phase6_coverage/REPORT.md`](../../ablations/phase6_coverage/REPORT.md).

## Goal
Decide how Phase 6 handles the 24% of queries that can never be covered, and see what each alpha costs. Methods: A (conditional), C (infinite), and B (C with a larger alpha).

## Setup
- Fit + calibration only (425 queries), 500 random re-splits into 75 calibration and 350 evaluation queries. **No test data.**
- One global budget (marginal split conformal). No signals yet.
- alpha 0.05 to 0.50.

## What happened (headline)
- A, alpha 0.1: conditional coverage 0.908 (target 0.90), unconditional 0.693, saves 45.1% vs k = 20.
- C, alpha ≤ 0.2: `q_hat` is infinite in 89-100% of re-splits, so savings are 0-2%.
- Fixed k = 11 on the same queries: conditional coverage 0.920, 1,355 tokens (A at alpha 0.1: 0.908, 1,399 tokens).

## Good findings
- Conformal hits its target coverage on average at every alpha.

## Surprising
- C is useless at the planned alpha (0% savings).
- With one global budget, A and C lie on the same trade-off curve; only the meaning of alpha differs.

## Stagnant / weak
- **A single global token budget is no better than fixed-k.** The gain must come from the per-query signals (Phase 5).
- Single splits can miss the target (p05 conditional coverage 0.838 at alpha 0.1).

## What went wrong
- `inf <= inf` bug made C look 100% covered. Found from an impossible number, fixed, and rerun before reporting.

## Next
- Recommendation: A, alpha 0.1. Add the global budget as a Phase 7 baseline. Princess decides.
