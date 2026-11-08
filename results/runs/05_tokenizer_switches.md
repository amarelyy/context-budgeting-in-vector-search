# Run 05: Tokenizer switches for `token_cost`

## Goal
`token_cost` should count tokens the same way the answering LLM does. The tokenizer changed as the LLM decision was made.

## Setup
1. `sentence-transformers/all-MiniLM-L6-v2` (first choice, before any LLM was chosen)
2. `Qwen/Qwen2.5-7B-Instruct` (after deciding to use a local Qwen LLM)
3. `Qwen/Qwen3-4B-Instruct-2507` (after the final model choice)

Each time: `python src\chunking.py`, then compare with the previous `chunks.jsonl`.

## What happened

| Tokenizer | median | mean | max | corpus total |
|---|---|---|---|---|
| MiniLM | 107.0 | 119.7 | 1846 | 593,478 |
| Qwen2.5 | 117.5 | 131.4 | 1985 | 651,022 |
| Qwen3 | 117.5 | 131.4 | 1985 | 651,022 |

- The chunk order and text stayed identical every time, so the embeddings and the index stayed valid.
- Qwen2.5 → Qwen3: **0 of 4,956 chunks changed `token_cost`.**

## Good findings
- Switching the tokenizer never forced a rebuild of the index. Only `token_cost` moved.

## Surprising
- **Qwen2.5 and Qwen3 count exactly the same.** They share the same vocabulary, so the baseline and oracle numbers did not need a rerun after the Qwen3 switch.
- Qwen counts about 10% more tokens than MiniLM on the same text.

## Stagnant / weak
- Recall is unaffected by the tokenizer (as expected, see run 03).

## What went wrong
- Nothing broke. The first choice (MiniLM) was a placeholder before the LLM was decided. It was labeled a proxy from the start.

## Next
- Token counts in all later runs use the Qwen3 tokenizer.
