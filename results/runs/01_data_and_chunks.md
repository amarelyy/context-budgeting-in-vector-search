# Run 01: Data, splits and chunks (tasks 1.1-1.3)

## Goal
Get a small HotpotQA subset with gold labels, split it into fit / calibration / test, and turn the passages into chunks with a `token_cost`.

## Setup
- Dataset: HotpotQA, distractor setting, validation set (7,405 questions), via `datasets` (`hotpotqa/hotpot_qa`).
- Sample: 500 queries, seed 42 (starting value).
- Split: 70 / 15 / 15 = 350 / 75 / 75 (starting value).
- Chunk = one paragraph. All queries share one corpus. Identical paragraphs (same title and text) become one chunk.
- Commands: `python src\data_loader.py`, then `python src\chunking.py`.
- Outputs: `data/raw/hotpot_subset.jsonl`, `data/processed/splits.json`, `data/processed/chunks.jsonl`, `data/processed/gold_chunk_ids.json`.

## What happened
- Download worked without an API key.
- Every HotpotQA question has 10 paragraphs: 2 gold (supporting) and 8 distractors.
- 500 queries × 10 paragraphs = 5,000 paragraphs, which became **4,956 chunks** after removing copies.
- `token_cost` (first run, MiniLM tokenizer): min 11, median 107, mean 119.7, max 1,846. Corpus total 593,478 tokens.
- Gold per query: exactly 2 for every query. No query had 0 gold.

## Good findings
- The gold labels link cleanly to chunks: every query has exactly 2 gold chunks. This means the label pipeline (Phase 4) starts from clean data.
- `token_cost` varies a lot (11 to 1,846). This is exactly why the project budgets in tokens instead of counting passages: 5 chunks can cost 55 or 9,000 tokens.

## Surprising
- Only 44 duplicate copies were merged (5,000 → 4,956). The distractor setting mostly gives each question its own paragraphs, so the shared corpus is only a little smaller than the raw total.

## Stagnant / weak
- Nothing yet. This run only builds data.

## What went wrong
- **Wrong working folder.** The first `chunking.py` run was started from inside `src/` and failed with `FileNotFoundError` for `data/raw/hotpot_subset.jsonl`. The paths are relative to the repo root. Fixed by running from the root. Rule since then: run every command from the repo root.
- **Harmless warning.** `Token indices sequence length is longer than the specified maximum sequence length (579 > 512)`. The tokenizer was only counting tokens, not feeding a model, so nothing broke. It does hint at a real issue later: the embedding model cuts long chunks (run 02).

## Next
- The tokenizer choice here was later changed twice (run 05). The chunks themselves never changed.
- The calibration and test splits are small (75 each), so `ece` and `coverage` will be noisy (Risk 4).
