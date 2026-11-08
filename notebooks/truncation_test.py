"""Task 2.3 Step 4: truncation guard test.

1. Our own count = Qwen3 tokenizer + chat template on the exact messages we send.
2. Compare with Ollama's prompt_eval_count.
3. Send the SAME prompt twice: does prompt caching lower prompt_eval_count?
4. Send a prompt that is too long for a small num_ctx: does Ollama error, or silently drop tokens?
Uses FIT queries only. Run from the repo root.
"""
import json
import sys

sys.path.insert(0, "src")
from transformers import AutoTokenizer  # noqa: E402

from llm import chat, MAIN_OPTIONS  # noqa: E402
from prompt import build_messages  # noqa: E402
from data_loader import load_raw  # noqa: E402
from chunking import TOKENIZER_NAME  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def own_count(messages):
    # newer transformers returns a dict; take the token list explicitly
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  return_dict=True)
    return len(out["input_ids"])


chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
ret = json.load(open("data/processed/retrieval_top20.json"))
splits = json.load(open("data/processed/splits.json"))
rows = {r["id"]: r for r in load_raw()}

q = splits["fit"][0]
passages = [chunks[c]["text"] for c in ret[q]["ids"][:5]]
msgs = build_messages(rows[q]["question"], passages)
ours = own_count(msgs)
sum_cost = sum(chunks[c]["token_cost"] for c in ret[q]["ids"][:5])

opts = dict(MAIN_OPTIONS, num_ctx=4096)
print("=== A: normal prompt (k=5), sent twice ===")
for i in (1, 2):
    r = chat(msgs, opts)
    print(f"call {i}: ours = {ours} | prompt_eval_count = {r['prompt_eval_count']} | "
          f"cached = {r.get('prompt_eval_cached_count')} | answer = {r['message']['content']!r}")
print(f"template overhead = ours - sum(token_cost) = {ours} - {sum_cost} = {ours - sum_cost}")

print("\n=== B: too long (k=20 text repeated) with num_ctx = 1024 ===")
long_passages = [chunks[c]["text"] for c in ret[q]["ids"][:20]]
long_msgs = build_messages(rows[q]["question"], long_passages)
ours_long = own_count(long_msgs)
try:
    r = chat(long_msgs, dict(MAIN_OPTIONS, num_ctx=1024))
    print(f"ours = {ours_long} | prompt_eval_count = {r['prompt_eval_count']} | "
          f"cached = {r.get('prompt_eval_cached_count')} | done_reason = {r['done_reason']} | "
          f"answer = {r['message']['content']!r}")
except Exception as e:  # noqa: BLE001
    print("ERROR from Ollama:", repr(e), getattr(e, "read", lambda: b"")()[:300])
