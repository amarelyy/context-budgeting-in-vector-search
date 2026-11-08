# Run 04: Oracle upper bound

## Goal
Find the maximum possible token saving before building the method. The oracle "cheats": it knows where the gold is and stops adding chunks exactly when the gold is covered. No real method can beat it.

## Setup
- **Fit split only** (350 queries). Rule from Princess: the oracle is never computed on test.
- Candidate pool: top 20 (same as the largest fixed k). Qwen tokenizer.
- Command: `python notebooks\oracle_fit.py`.

## What happened

| | tokens (mean) | tokens (median) |
|---|---|---|
| fixed k = 20, all fit queries | 2558.4 | |
| oracle, stop when **both** gold are covered | 591.5 | 349.0 |
| oracle, stop when the **first** gold is covered | 143.2 | 96.0 |

- Both gold in the top 20: 267 of 350 queries.
- **Maximum possible saving vs k = 20 (both gold, same 267 queries): 76.7%.**

## Good findings
- **There is a lot of room.** Even a method that gets only part of the way to the oracle saves many tokens.
- **Cost per query varies a lot.** The oracle median (349) is far below its mean (591.5). Some queries need 1-2 chunks, others need many. One fixed k cannot fit both kinds, which supports a per-query budget.

## Surprising
- The first gold usually costs under 100 tokens (median 96), meaning it is often the very first chunk.

## Stagnant / weak
- 83 of 350 fit queries (24%) never get both gold inside the top 20. For them, no budget helps.

## What went wrong
- **The oracle was first computed on the test split by mistake** (58 of 75 queries, about 78% ceiling). Princess then set the rule "oracle on fit only". The test oracle was deleted from the log. Nothing was tuned from it. It is mentioned here only so the record is honest.

## Next
- Phase 8 reports our method's saving next to this 76.7% ceiling, for example "we reach X% of the maximum possible saving".
