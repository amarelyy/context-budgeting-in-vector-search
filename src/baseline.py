"""Phase 2 (2.1, 2.2): fixed-k baseline on the TEST split.

For each test query, retrieve the top k chunks (k = 5, 10, 20) and measure:
- recall_at_k (both):  share of queries where BOTH gold chunks are in the top k
- recall_at_k (any):   share of queries where AT LEAST ONE gold chunk is in the top k
- gold_frac:           mean share of gold chunks found (0, 0.5 or 1 per query)
- total_tokens:        sum of token_cost of the k chunks fed to the LLM (mean over queries)
"""
import json

import faiss
import numpy as np

from data_loader import load_raw
from index import INDEX_PATH, load_chunks, get_model, embed_texts, search_vec

KS = (5, 10, 20)
SPLIT = "test"
OUT_PATH = "data/processed/baseline_test_top20.json"


def evaluate(I, qids, gold, token_cost, k):
    both, any_, frac, toks = [], [], [], []
    for i, q in enumerate(qids):
        top = set(I[i, :k].tolist())
        g = set(gold[q])
        hit = len(g & top)
        both.append(hit == len(g))
        any_.append(hit > 0)
        frac.append(hit / len(g))
        toks.append(int(token_cost[I[i, :k]].sum()))
    return {
        "k": k,
        "recall_at_k_both": float(np.mean(both)),
        "recall_at_k_any": float(np.mean(any_)),
        "gold_frac": float(np.mean(frac)),
        "total_tokens_mean": float(np.mean(toks)),
        "total_tokens_median": float(np.median(toks)),
    }


if __name__ == "__main__":
    chunks = load_chunks()
    token_cost = np.array([c["token_cost"] for c in chunks])
    index = faiss.read_index(INDEX_PATH)
    model = get_model()
    splits = json.load(open("data/processed/splits.json"))
    gold = json.load(open("data/processed/gold_chunk_ids.json"))
    rows = {r["id"]: r for r in load_raw()}

    qids = splits[SPLIT]
    q_vecs = embed_texts(model, [rows[q]["question"] for q in qids])
    I, D = search_vec(index, q_vecs, max(KS))  # retrieve once, slice for each k

    # save the ranked lists so later phases compare on the same retrieval
    with open(OUT_PATH, "w") as f:
        json.dump({q: {"ids": I[i].tolist(), "dists": D[i].round(6).tolist()}
                   for i, q in enumerate(qids)}, f)

    print(f"Fixed-k baseline | split = {SPLIT} ({len(qids)} queries)")
    print(f"{'k':>3} | {'recall both':>11} | {'recall any':>10} | {'gold_frac':>9} | "
          f"{'tokens mean':>11} | {'tokens median':>13}")
    for k in KS:
        r = evaluate(I, qids, gold, token_cost, k)
        print(f"{k:>3} | {r['recall_at_k_both']:>11.3f} | {r['recall_at_k_any']:>10.3f} | "
              f"{r['gold_frac']:>9.3f} | {r['total_tokens_mean']:>11.1f} | "
              f"{r['total_tokens_median']:>13.1f}")
