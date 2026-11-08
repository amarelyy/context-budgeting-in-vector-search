"""Oracle upper bound on the FIT split only (never on test).

Oracle = stop adding chunks exactly when the gold is covered. It uses the labels,
so no real method can do better. It is the maximum possible token saving.
Run from the repo root:  .\.venv\Scripts\python.exe notebooks\oracle_fit.py
"""
import json
import sys

import faiss
import numpy as np

sys.path.insert(0, "src")
from data_loader import load_raw  # noqa: E402
from index import INDEX_PATH, load_chunks, get_model, embed_texts, search_vec  # noqa: E402

TOP_M = 20  # same candidate pool as the largest fixed k

chunks = load_chunks()
tc = np.array([c["token_cost"] for c in chunks])
index = faiss.read_index(INDEX_PATH)
model = get_model()
splits = json.load(open("data/processed/splits.json"))
gold = json.load(open("data/processed/gold_chunk_ids.json"))
rows = {r["id"]: r for r in load_raw()}

qids = splits["fit"]
I, _ = search_vec(index, embed_texts(model, [rows[q]["question"] for q in qids]), TOP_M)

need_both, need_first, k20_both, k20_all = [], [], [], []
for i, q in enumerate(qids):
    ids = I[i].tolist()
    cum = np.cumsum(tc[ids])
    pos = [ids.index(g) for g in gold[q] if g in ids]
    k20_all.append(cum[-1])
    if pos:
        need_first.append(cum[min(pos)])
    if len(pos) == len(gold[q]):
        need_both.append(cum[max(pos)])
        k20_both.append(cum[-1])  # k=20 cost on the SAME queries

print(f"Oracle on FIT split ({len(qids)} queries, top {TOP_M})")
print(f"both gold in top {TOP_M}: {len(need_both)} of {len(qids)}")
print(f"k=20 tokens, all queries: mean {np.mean(k20_all):.1f}")
print(f"oracle tokens to cover both gold: mean {np.mean(need_both):.1f} | median {np.median(need_both):.1f}")
print(f"oracle tokens to cover first gold: mean {np.mean(need_first):.1f} | median {np.median(need_first):.1f}")
print(f"max saving vs k=20 (both gold, same {len(need_both)} queries): "
      f"{100 * (1 - np.mean(need_both) / np.mean(k20_both)):.1f}%")
