# Run 25: Index-time deduplication (Phase 9.4)

Full write-up: [`ablations/PHASE9_REPORT.md`](../../ablations/PHASE9_REPORT.md). Data: `ablations/phase9_dedup/`.

## Goal
Remove near-duplicate chunks at index time so the budget is not wasted on copies, rebuild the index, rerun Phase 7-8, and compare.

## Setup
- `python src\dedup.py`. Step 1 reports near-duplicate pairs by embedding cosine at 0.90 / 0.95 / 0.98 / 0.99. Step 2 applies the final rule, rebuilds HNSW (same parameters) without the dropped chunks, re-retrieves top 20 for calibration and test, remaps gold to the kept copy, recomputes labels, `q_hat` and the Phase 7-8 retrieval numbers. The LLM was not rerun.

## What happened
- By embedding similarity alone: 0.90 → 62 chunks dropped (7 gold); 0.95 → 28 (3 gold); 0.98 → 11 (2 gold); 0.99 → 10 (2 gold).
- **At 0.95 the "duplicates" are different entities** with similar wording (VCU Rams 2012-13 vs 2013-14 seasons, Miss Nepal 2013 vs 2014, Sunset at Chaophraya 1996 vs 2013 film).
- At ≥ 0.99 most pairs are true copies: **identical text under different titles** (6 "Tom Clancy's Net Force Explorers" book pages, "Kim Ryeowook" vs "Kim Ryeowook discography", Cromwell disambiguation pages, Colomac Mine). One look-alike remains (2008 Swedish Open vs Hypo Group Tennis, text similarity 0.919).
- **Final rule: cos ≥ 0.99 AND text similarity ≥ 0.98** → 9 chunks dropped (0.18%). 4 of 75 test top-20 lists change.

| test | before dedup | after dedup |
|---|---|---|
| global budget: recall / cond. coverage / tokens | 0.720 / 0.931 / 1,551 | 0.733 / 0.932 / 1,546 |
| fixed k = 12 (calibration-chosen) | 0.760 / 0.983 / 1,543 | 0.773 / 0.983 / 1,549 |
| k = 20: both gold in pool | 0.773 | 0.787 |
| calibration `q_hat` | 1,649 | 1,649 |

## Good findings
- Dedup helps slightly: one more test query gets both gold within reach. Nothing gets worse.

## Surprising
- **Embedding similarity is a bad duplicate detector for factual corpora.** Pages about different seasons, years or films look 95%+ similar to the embedding model but hold different facts. Merging them would delete gold passages. Text similarity is needed as a second check.
- The real duplicates came from chunking: 1.3 merged only same-title-and-text chunks, so identical text under different titles slipped through.

## Stagnant / weak
- The effect is tiny on HotpotQA (0.18% of chunks); the method ranking is unchanged.

## What went wrong
- The first idea (embedding cos ≥ 0.95 alone) would have removed 3 gold chunks. Found by inspecting the pairs before applying anything.
