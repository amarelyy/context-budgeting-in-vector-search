# Run 26: Transfer to BEIR SciFact (Phase 9.5, stretch)

Full write-up: [`ablations/PHASE9_REPORT.md`](../../ablations/PHASE9_REPORT.md). Data: `ablations/phase9_transfer/`.

## Goal
Do the HotpotQA settings (`q_hat`, k, CQR) still give ~90% coverage on a different dataset without re-calibration? (Risk 2 / CONTEXT §6 caveat.)

## Setup
- **Dataset: BEIR SciFact** (starting choice; the BEIR subset was an open question). 5,183 abstracts, 300 test queries with qrels (277 have 1 relevant doc, 23 have 2-5).
- Same pipeline: one chunk = one abstract's text, Qwen3 `token_cost`, MiniLM embeddings, HNSW (same parameters), top 20. Covered = all relevant docs fed. Retrieval only (no short answers, so no LLM).
- A: HotpotQA settings as they are. B: re-calibrated on SciFact, 500 re-splits (150 calibrate / 150 evaluate), alpha 0.1, method A.
- `python ablations\phase9_transfer\run_scifact.py`

## What happened
- SciFact `token_cost`: median 281 (HotpotQA 117.5), mean 308, max 1,911; 58% of abstracts exceed MiniLM's 256-token limit. All relevant in top 20: 0.823. k = 20 costs 6,185 tokens.

| method | cond. coverage (target 0.90) | tokens | avg chunks | saved vs k=20 |
|---|---|---|---|---|
| A: HotpotQA token budget (1,649) | **0.822** (under) | 1,452 | 5.2 | 76.5% |
| A: HotpotQA fixed k = 12 | **0.960** (over) | 3,709 | 12 | 40.0% |
| A: HotpotQA CQR | **0.814** (under) | 1,221 | 4.3 | 80.3% |
| B: re-calibrated token budget | 0.906 | 2,587 | | 58.1% |
| B: re-calibrated conformal k (median k = 8) | 0.913 | 2,511 | 8 | 59.4% |
| B: re-calibrated CQR | 0.904 | 2,669 | | 56.8% |

## Good findings
- **Re-calibration with only 150 queries restores the guarantee** for every method (0.904-0.913).

## Surprising
- **Without re-calibration, the guarantee breaks, and the two units break in opposite directions.** SciFact chunks are ~2.4× longer: 1,649 tokens now buys only ~5 chunks (under-coverage, 0.822), while 12 chunks now cost 3,709 tokens (over-payment, 0.960 coverage). Neither unit is safe under a shift in chunk length.
- After re-calibration, chunks and tokens again end up equivalent (2,511 vs 2,587 tokens).

## Stagnant / weak
- No method beats conformal-k on SciFact either.

## What went wrong
- Nothing broke. Limits: one BEIR dataset; mostly single-relevant queries, so "all relevant covered" is an easier target than HotpotQA's "both gold".
