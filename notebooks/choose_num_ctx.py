"""Task 2.3 Step 5: choose num_ctx from real data.

num_ctx must be larger than the biggest prompt we will ever send (k = 20, the largest
pool; the token-budget method picks from the same top 20) plus num_predict.
Lengths only (no labels, no tuning), so all splits are used.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from data_loader import load_raw  # noqa: E402
from llm import count_prompt_tokens, MAIN_OPTIONS  # noqa: E402
from prompt import build_messages  # noqa: E402

chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
ret = json.load(open("data/processed/retrieval_top20.json"))
rows = {r["id"]: r for r in load_raw()}

lens, overhead = [], []
for q, v in ret.items():
    texts = [chunks[c]["text"] for c in v["ids"][:20]]
    n = count_prompt_tokens(build_messages(rows[q]["question"], texts))
    lens.append(n)
    overhead.append(n - sum(chunks[c]["token_cost"] for c in v["ids"][:20]))
lens, overhead = np.array(lens), np.array(overhead)

need = lens.max() + MAIN_OPTIONS["num_predict"]
num_ctx = int(np.ceil(need / 1024) * 1024)
print(f"k=20 prompt tokens over {len(lens)} queries: min {lens.min()} | median {np.median(lens):.0f} | "
      f"p99 {np.percentile(lens, 99):.0f} | max {lens.max()}")
print(f"overhead (prompt - sum token_cost), k=20: min {overhead.min()} | median {np.median(overhead):.0f} | max {overhead.max()}")
k0 = [count_prompt_tokens(build_messages(rows[q]["question"], [])) for q in list(ret)[:50]]
print(f"k=0 prompt tokens (50 queries): median {np.median(k0):.0f}")
print(f"max prompt + num_predict ({MAIN_OPTIONS['num_predict']}) = {need} -> num_ctx = {num_ctx} (rounded up to 1024)")
