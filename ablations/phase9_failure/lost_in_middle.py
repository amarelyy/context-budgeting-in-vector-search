"""Phase 9.3: controlled "lost in the middle" test (fit + calibration only, no test data).

For every coverable fit/calibration query (both gold in the top 20), build the SAME 20-chunk
context (2 gold + 18 retrieved distractors) and only move the gold pair:
  start  : gold at positions 1-2
  middle : gold at positions 10-11
  end    : gold at positions 19-20
plus a reference:
  gold_only : just the 2 gold chunks
Gold keeps its ranked order; distractors keep theirs. Tokens are identical across start/middle/end,
so any F1 difference comes from POSITION alone. LLM main setting (temperature 0, seed 42).
Output: ablations/phase9_failure/lost_in_middle.csv
"""
import csv
import json
import sys
import time

import pandas as pd

sys.path.insert(0, "src")
from data_loader import load_raw  # noqa: E402
from llm import chat_checked, MAIN_OPTIONS  # noqa: E402
from prompt import build_messages  # noqa: E402
from qa_metrics import exact_match, f1_score  # noqa: E402

OUT = "ablations/phase9_failure/lost_in_middle.csv"

chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
ret = json.load(open("data/processed/retrieval_top20.json"))
gold = json.load(open("data/processed/gold_chunk_ids.json"))
lab = pd.read_csv("data/processed/labels.csv")
rows = {r["id"]: r for r in load_raw()}
qids = lab[lab.split.isin(["fit", "calibration"]) & lab.gold_in_pool].query_id.tolist()

recs, t0 = [], time.time()
for q in qids:
    ids = ret[q]["ids"]
    g = [c for c in ids if c in gold[q]]          # gold in ranked order
    o = [c for c in ids if c not in gold[q]]      # 18 distractors in ranked order
    layouts = {"start": g + o, "middle": o[:9] + g + o[9:], "end": o + g, "gold_only": g}
    for name, ctx in layouts.items():
        resp, info = chat_checked(build_messages(rows[q]["question"], [chunks[c]["text"] for c in ctx]),
                                  MAIN_OPTIONS)
        ans = resp["message"]["content"].strip()
        recs.append({"query_id": q, "layout": name, "n_chunks": len(ctx),
                     "gold_positions": " ".join(str(ctx.index(x) + 1) for x in g),
                     "truncated": info["truncated"], "answer": ans, "gold_answer": rows[q]["answer"],
                     "em": exact_match(ans, rows[q]["answer"]), "f1": round(f1_score(ans, rows[q]["answer"]), 4)})
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
d = pd.DataFrame(recs)
print(f"{len(qids)} queries x 4 layouts = {len(d)} calls in {time.time() - t0:.0f} s | truncated {int(d.truncated.sum())}")
print(d.groupby("layout")[["em", "f1"]].mean().round(3).reindex(["gold_only", "start", "middle", "end"]).to_string())
