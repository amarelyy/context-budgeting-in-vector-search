# Run 09: Truncation guard test (task 2.3 Step 4)

## Goal
Third-party sources reported that Ollama silently drops the start of a prompt that is too long. Test whether this happens on Princess's setup, and build a guard that would catch it.

## Setup
- One fit query, k = 5 chunks, the project prompt template (`src/prompt.py`).
- Our count: Qwen3 tokenizer + chat template on the exact messages. Ollama's count: `prompt_eval_count`.
- Test A: the same prompt sent twice (does caching change the count?).
- Test B: a k = 20 prompt with num_ctx 1024 (too long on purpose).
- Command: `python notebooks\truncation_test.py`.

## What happened
- Test A: our count **485**, Ollama **485**, both calls. Second call: `prompt_eval_cached_count` 484, but `prompt_eval_count` still 485. Answer "Genesee River" both times.
- Template overhead for this k = 5 prompt: 485 − 402 (sum of `token_cost`) = **83 tokens**.
- Test B: **HTTP 400**, `"request (2833 tokens) exceeds the available context size (1024 tokens), try increasing it"`.

## Good findings
- **Our token count matches Ollama exactly.** Our tokenizer + chat template rebuild the prompt the same way Ollama does.
- **The prompt cache does not lower `prompt_eval_count`.** The guard will not raise false alarms.
- Same answer on both calls at temperature 0.

## Surprising
- **No silent truncation on Ollama 0.40.15.** A too-long prompt is refused with an error. The third-party reports were not reproduced here (they may describe other versions or the `/v1` endpoint). The guard stays in the code anyway.

## Stagnant / weak
- Nothing.

## What went wrong
- **Bug in our own count.** The first run printed "ours = 2". In this `transformers` version, `apply_chat_template(..., tokenize=True)` returns a dict, and the code measured the dict's length (2 keys) instead of the token list. Fixed with `return_dict=True` and `["input_ids"]`. The rerun gave 485 = 485.

## Next
- The guard lives in `llm.chat_checked`. It refuses to send if prompt + num_predict > num_ctx, and flags `truncated` if Ollama reads fewer tokens than we sent.
