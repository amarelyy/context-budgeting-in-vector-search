# Run 17: Conformal budget per bin (Phase 6, tasks 6.1-6.3) + dev estimate

Full dev-estimate report: [`ablations/phase6_mondrian/REPORT.md`](../../ablations/phase6_mondrian/REPORT.md).

## Goal
Compute `q_hat` on the calibration split: one global and one per `conf_score` bin (Mondrian). Check coverage. Then estimate fairly, without test, whether bins save tokens.

## Setup
- alpha 0.1, method A (coverable calibration queries only), finite-sample correction.
- **3 bins (starting value)**, edges = tertiles of `conf_score` on **fit**: 0.3423 and 0.4191.
- Commands: `python src\conformal.py` (6.1-6.3, saves `data/processed/conformal.json`), then `python ablations\phase6_mondrian\run_mondrian.py` (dev estimate: 500 re-splits of fit + calibration, out-of-fold scores).

## What happened
Real calibration split:

| | calibration n | coverable n | median tokens needed | q_hat |
|---|---|---|---|---|
| global | 75 | 57 | | 1,649 |
| bin 0, low confidence | 28 | 17 | 645 | 2,097 |
| bin 1, mid | 22 | 20 | 399 | 1,731 |
| bin 2, high | 25 | 20 | 288 | 950 |

- 6.3 coverage on calibration (in-sample, optimistic by design): bins 0.965 conditional / 0.733 unconditional; global 0.930 conditional. Mean budget: bins 1,607 vs global 1,649.

Dev estimate (500 re-splits, no test): global 1,399 tokens (45.1% saved), 2 bins 1,507 (40.9%), 3 bins 1,614 (36.7%). Fixed k = 12: 0.929 coverage with 1,486 tokens.

## Good findings
- **The score sorts queries in the right order.** Low-confidence queries need more tokens (median 645) than high-confidence ones (288).
- Coverage reaches the target in every variant.

## Surprising
- **More bins = more tokens.** The bins are cheaper than the global budget in only 16-24% of re-splits.

## Stagnant / weak
- **The planned method does not beat the global budget or fixed-k** on dev data. Cause: each small bin pays its own finite-sample penalty (bins of 14-24 coverable queries over-cover at 0.92-0.97), and the weak score (AUROC 0.59) cannot make up for it.

## What went wrong
- Nothing broke. The method works as designed; the setup (weak signals + 75 calibration queries) is the limit.

## Next
- **Stop and discuss with Princess before Phase 7.** Options in the report: (1) more data, (2) conformalized quantile regression instead of bins, (3) a stronger signal.
