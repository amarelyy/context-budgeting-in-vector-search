"""Task 2.4: fixed-k baseline WITH the LLM, on the TEST split.

For each test query and each k in (0, 5, 10, 20): feed the top-k chunks to the LLM with the
one prompt template, then score the answer (EM, F1). One CSV row per query per k.

Usage (repo root):
  python src\llm_baseline.py main            # temperature 0, seed 42 (headline numbers)
  python src\llm_baseline.py qwen 1 2 3      # Qwen's values, one run per seed (error analysis)
"""
import csv
import json
import sys
import time

import numpy as np

from data_loader import load_raw
from llm import chat_checked, MAIN_OPTIONS, QWEN_OPTIONS, MODEL
from prompt import build_messages
from qa_metrics import exact_match, f1_score

KS = (0, 5, 10, 20)
SPLIT = "test"
FIELDS = ["query_id", "split", "method", "k", "setting", "seed", "num_ctx", "num_predict",
          "fed_tokens", "prompt_tokens_ours", "prompt_eval_count", "truncated", "done_reason",
          "gold_covered_both", "gold_covered_any", "gold_ranks", "answer", "gold_answer",
          "em", "f1", "latency_s"]


def run(setting, seed, options, qids, rows, chunks, ret, gold):
    out_path = f"results/llm_baseline_{SPLIT}_{setting}_seed{seed}.csv"
    opts = dict(options, seed=seed)
    records = []
    for q in qids:
        ids = ret[q]["ids"]
        for k in KS:
            fed = ids[:k]
            msgs = build_messages(rows[q]["question"], [chunks[c]["text"] for c in fed])
            t0 = time.time()
            resp, info = chat_checked(msgs, opts)
            ans = resp["message"]["content"].strip()
            g = set(gold[q])
            ranks = [ids.index(x) + 1 for x in gold[q] if x in fed]
            records.append({
                "query_id": q, "split": SPLIT, "method": "fixed_k", "k": k,
                "setting": setting, "seed": seed, "num_ctx": opts["num_ctx"],
                "num_predict": opts["num_predict"],
                "fed_tokens": int(sum(chunks[c]["token_cost"] for c in fed)),
                **info,
                "gold_covered_both": g <= set(fed), "gold_covered_any": bool(g & set(fed)),
                "gold_ranks": " ".join(map(str, sorted(ranks))),
                "answer": ans, "gold_answer": rows[q]["answer"],
                "em": exact_match(ans, rows[q]["answer"]),
                "f1": round(f1_score(ans, rows[q]["answer"]), 4),
                "latency_s": round(time.time() - t0, 3),
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    return out_path, records


def summarize(records):
    print(f"{'k':>3} | {'recall both':>11} | {'fed tokens':>10} | {'EM':>6} | {'F1':>6} | "
          f"{'truncated':>9} | {'cut by num_predict':>18}")
    for k in KS:
        r = [x for x in records if x["k"] == k]
        print(f"{k:>3} | {np.mean([x['gold_covered_both'] for x in r]):>11.3f} | "
              f"{np.mean([x['fed_tokens'] for x in r]):>10.1f} | {np.mean([x['em'] for x in r]):>6.3f} | "
              f"{np.mean([x['f1'] for x in r]):>6.3f} | {sum(x['truncated'] for x in r):>9} | "
              f"{sum(x['done_reason'] != 'stop' for x in r):>18}")


if __name__ == "__main__":
    setting = sys.argv[1] if len(sys.argv) > 1 else "main"
    chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
    ret = json.load(open("data/processed/retrieval_top20.json"))
    gold = json.load(open("data/processed/gold_chunk_ids.json"))
    qids = json.load(open("data/processed/splits.json"))[SPLIT]
    rows = {r["id"]: r for r in load_raw()}

    if setting == "main":
        runs = [(MAIN_OPTIONS["seed"], MAIN_OPTIONS)]
    else:
        runs = [(int(s), QWEN_OPTIONS) for s in sys.argv[2:]]
    for seed, options in runs:
        t0 = time.time()
        path, recs = run(setting, seed, options, qids, rows, chunks, ret, gold)
        print(f"\n[{setting}, seed {seed}] model {MODEL} | {len(qids)} {SPLIT} queries | "
              f"num_ctx {options['num_ctx']} | {time.time() - t0:.0f} s | saved {path}")
        summarize(recs)
