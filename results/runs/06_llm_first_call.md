# Run 06: First LLM call (task 2.3 Step 3)

## Goal
Check that Python can call the local LLM through Ollama, see the raw response, and confirm that our settings are really applied.

## Setup
- Model: `qwen3:4b-instruct-2507-q4_K_M` (ID `0edcdef34593`, Q4_K_M), Ollama 0.40.15.
- Native `/api/chat`, all options sent explicitly: temperature 0, seed 42, num_ctx 4096 (placeholder), num_predict 256.
- Test prompt: "Context: Paris is the capital of France. Question: What is the capital of France? Answer with a short answer only."
- Command: `python src\llm.py`, then `ollama ps`.

## What happened
- Answer: `"Paris"`, `done_reason` = `"stop"`.
- `prompt_eval_count` 33, `eval_count` 2.
- `load_duration` 4.2 s (first load only), `eval_duration` 0.02 s.
- `ollama ps`: CONTEXT **4096**, PROCESSOR **100% GPU**, SIZE 3.2 GB.

## Good findings
- **The `num_ctx` we send is really applied** (ollama ps shows 4096).
- The model fits fully on the GPU with room to spare in 8 GB VRAM.
- No `<think>` block in the answer.

## Surprising
- `ollama show` lists a "thinking" capability for this non-thinking model, but no thinking text appeared. It seems to be a label for the Qwen3 family.
- For this model, Ollama's built-in temperature is **0.7** (Qwen's value), not 0. Temperature 0 must always be passed on purpose.
- A ~20-word prompt counted as 33 tokens. The extra tokens are the chat template wrappers.

## Stagnant / weak
- Nothing to compare yet.

## What went wrong
- Nothing. (Princess's Ollama models were only read, never changed.)

## Next
- Measure the chat template overhead and test truncation (run 09).
