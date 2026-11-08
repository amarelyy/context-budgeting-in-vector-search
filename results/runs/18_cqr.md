# Run 18: Conformalized quantile regression, CQR (Phase 6, option 2)

Full report: [`ablations/phase6_methods/REPORT.md`](../../ablations/phase6_methods/REPORT.md).

## Goal
Give each query its own budget without splitting calibration into small bins (the cause of the bins' failure in run 17).

## Setup
- `src/cqr.py`: linear quantile regression (quantile 0.9, no penalty) of log(`gold_pos_tokens`) on the 5 signals, fitted on coverable fit queries. One conformal correction Q from all coverable calibration queries. Budget = exp(prediction + Q). alpha 0.1, method A.
- Dev estimate: `ablations/phase6_methods/run_methods.py` (out-of-fold, 500 re-splits, no test).

## What happened
- Real calibration (in-sample): Q = +0.143, budgets 1,028-2,412 (mean 1,503 vs global 1,649), conditional coverage 0.912.
- Dev estimate: CQR 0.908 coverage with 1,505 tokens (41.0% saved). Global: 0.908 with 1,399 (45.1%). 3 bins: 0.923 with 1,614 (36.7%). CQR is cheaper than global in 27% of re-splits.

## Good findings
- CQR hits the coverage target exactly (no over-coverage) and beats bins.

## Surprising
- Per-query budgets from a weak predictor cost **more** than one shared budget. Budgets that vary for the wrong reasons push the average up.

## Stagnant / weak
- Still no gain over the global budget.

## What went wrong
- Nothing broke.

## Next
- Option 3: add the question-type signal (run 19).
