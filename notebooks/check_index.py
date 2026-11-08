"""Phase 1 (1.6, 1.7): sanity check the index and compare HNSW with exact search.

Uses FIT queries only. The test split is not touched before Phase 7.
Run from the repo root:  .\.venv\Scripts\python.exe notebooks\check_index.py
"""
import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, "src")
from data_loader import load_raw  # noqa: E402
from index import (INDEX_PATH, EMB_PATH, load_chunks, get_model,  # noqa: E402
                   embed_texts, search_vec)

TOP_M = 20  # candidates for the recall check (starting value)

chunks = load_chunks()
emb = np.load(EMB_PATH)
index = faiss.read_index(INDEX_PATH)
model = get_model()
splits = json.load(open("data/processed/splits.json"))
gold = json.load(open("data/processed/gold_chunk_ids.json"))
rows = {r["id"]: r for r in load_raw()}

fit_ids = splits["fit"]
q_vecs = embed_texts(model, [rows[i]["question"] for i in fit_ids])

# ---- 1.6 sanity check: a few queries by hand ----
print("=== 1.6 sanity check (3 fit queries, top 5) ===")
I, D = search_vec(index, q_vecs[:3], 5)
for qi, qid in enumerate(fit_ids[:3]):
    print("\nQ:", rows[qid]["question"])
    print("   gold:", [chunks[c]["title"] for c in gold[qid]])
    for rank, (cid, d) in enumerate(zip(I[qi], D[qi]), 1):
        mark = "GOLD" if cid in gold[qid] else "    "
        print(f"   {rank}. d={d:.4f} {mark} {chunks[cid]['title']}")

# distance range, to check the distance type behaves as expected
print("\ndistance type: L2 on unit vectors (sqrt of FAISS squared L2), range [0, 2]")
I20, D20 = search_vec(index, q_vecs, TOP_M)
print("top-1 dist  min/median/max:", D20[:, 0].min().round(4),
      np.median(D20[:, 0]).round(4), D20[:, 0].max().round(4))

# how many chunks are longer than the model can read (they get truncated)
max_len = model.max_seq_length
n_trunc = sum(1 for c in chunks if c["token_cost"] > max_len)
print(f"chunks longer than model max_seq_length ({max_len}): {n_trunc} of {len(chunks)}")

# info only (not the baseline): share of fit queries with both / at least one gold in top m
for m in (5, 10, 20):
    both = np.mean([set(gold[q]) <= set(I20[i, :m]) for i, q in enumerate(fit_ids)])
    any_ = np.mean([bool(set(gold[q]) & set(I20[i, :m])) for i, q in enumerate(fit_ids)])
    print(f"fit, top {m:>2}: both gold = {both:.3f} | at least one gold = {any_:.3f}")

# ---- 1.7 HNSW vs exact search ----
print(f"\n=== 1.7 HNSW recall vs exact search (fit queries = {len(fit_ids)}, top {TOP_M}) ===")
exact = faiss.IndexFlatL2(emb.shape[1])
exact.add(emb)
_, I_exact = exact.search(q_vecs, TOP_M)
overlap = [len(set(I20[i]) & set(I_exact[i])) / TOP_M for i in range(len(fit_ids))]
print(f"recall@{TOP_M} of HNSW vs exact: mean = {np.mean(overlap):.4f} | min = {np.min(overlap):.4f}")
print("top-1 identical:", np.mean(I20[:, 0] == I_exact[:, 0]).round(4))
