# Run 10: Choosing num_ctx (task 2.3 Step 5)

## Goal
Pick a `num_ctx` that fits the largest prompt we will ever send, and use it for every method (a fair comparison needs the same value).

## Setup
- Largest prompt = k = 20 (the token-budget method picks from the same top 20, so it never needs more).
- All 500 queries, lengths only (no labels, nothing tuned).
- Command: `python notebooks\choose_num_ctx.py`.

## What happened
- k = 20 prompt tokens: min 1,484, median 2,670, p99 3,890, **max 4,599**.
- Overhead (prompt − sum of `token_cost`) at k = 20: min 116, median 136, max 166.
- k = 0 prompt: median 60 tokens.
- 4,599 + num_predict 256 = 4,855, so **num_ctx = 5120** (rounded up to a multiple of 1024, starting value).

## Good findings
- 5120 is small: it fits easily in 8 GB VRAM.
- Overhead is small and stable (~136 tokens), so `token_cost` alone is a good measure of what changes between methods.

## Surprising
- The Ollama default (4096) would have been **too small** for some k = 20 prompts (max 4,599). Ollama 0.40.15 would have refused them with an error (run 09), but this shows why `num_ctx` must be set on purpose.

## Stagnant / weak
- Nothing.

## What went wrong
- Nothing.

## Next
- If the corpus, chunking or prompt template changes, rerun this and update `num_ctx`.
