# Run 07: Signals and unit test (tasks 3.1-3.2)

## Goal
Write the five confidence signals and check them against the worked example from the UARP proposal.

## Setup
- `src/signals.py`, `top_m` = 20 (starting value), softmax temperature for entropy `tau` = 0.1 (starting value).
- `tests/test_signals.py`: plain asserts, no pytest.
- Commands: `python tests\test_signals.py`, then a demo on 3 fit queries.

## What happened
- 3 of 3 tests pass:
  - Worked example 0.20, 0.21, 0.90: mean 0.4367, `margin_mc` 0.01, `var_local` 0.1074, `top1_dist` 0.20, `elbow_idx` 2.
  - Unsorted input gives the same result.
  - Entropy: all-equal distances give 1.0; one clear winner gives about 0.
- Demo on 3 fit queries:

| Query | margin_mc | var_local | elbow_idx | entropy | gold ranks |
|---|---|---|---|---|---|
| Tamiami Canal / Genesee River | 0.0454 | 0.0050 | 3 | 0.86 | 1, 2 |
| Palm Springs | 0.2875 | 0.0271 | 1 | 0.20 | 1, 2 |
| Saab 36 / Avro Vulcan | 0.0070 | 0.0040 | 4 | 0.93 | 1, 2 |

## Good findings
- The formulas match the proposal exactly.

## Surprising
- **The Saab query has perfect retrieval (both gold at ranks 1-2) but the smallest margin (0.007).** The two gold passages are equally good, so they sit close together. In single-answer search, a small margin means "unsure". In 2-hop search it can mean "two good answers tied at the top". This was later confirmed on all fit queries (run 08).

## Stagnant / weak
- With `tau` = 1, entropy would be almost 1 for every query, because the distances only span ~0.4-1.1. That is why `tau` = 0.1 was chosen.

## What went wrong
- Nothing.

## Next
- Check the signals on all queries (run 08).
