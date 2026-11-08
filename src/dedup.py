"""Phase 9.4: index-time deduplication of near-duplicate chunks.

Two chunks are near-duplicates if the cosine similarity of their embeddings >= THRESHOLD
(starting value 0.95). For each near-duplicate group, keep the chunk with the lowest id and
drop the others. Gold labels are remapped to the kept chunk (a dropped gold chunk's copy
still carries the same content).

This script first REPORTS how many chunks would be dropped at several thresholds, and how many
retrieved top-20 lists (per split) contain a dropped chunk. If no top-20 list contains a dropped
chunk, rebuilding the index cannot change any Phase 7-8 result, and the rerun is skipped.
"""
import json

import numpy as np
import pandas as pd

THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
THRESHOLD = 0.95   # starting value

emb = np.load("data/processed/chunk_embeddings.npy")
chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
ret = json.load(open("data/processed/retrieval_top20.json"))
splits = json.load(open("data/processed/splits.json"))
gold = json.load(open("data/processed/gold_chunk_ids.json"))

sim = emb @ emb.T                       # unit vectors -> cosine similarity
np.fill_diagonal(sim, -1.0)
iu = np.triu_indices(len(emb), k=1)

rows = []
for th in THRESHOLDS:
    i, j = np.where(np.triu(sim >= th, k=1))
    dropped = set()
    for a, b in sorted(zip(i, j)):
        if a not in dropped:
            dropped.add(int(b))
    row = {"threshold": th, "pairs": len(i), "chunks_dropped": len(dropped),
           "dropped_share": round(len(dropped) / len(chunks), 4)}
    for s, qs in splits.items():
        row[f"{s}_top20_lists_affected"] = sum(bool(dropped & set(ret[q]["ids"])) for q in qs)
    row["gold_chunks_dropped"] = sum(len(dropped & set(v)) for v in gold.values())
    rows.append(row)
    if th == THRESHOLD:
        examples = [(int(a), int(b), round(float(sim[a, b]), 4)) for a, b in list(zip(i, j))[:5]]

out = pd.DataFrame(rows)
out.to_csv("ablations/phase9_dedup/dedup_report.csv", index=False)
print(out.to_string(index=False))
print(f"\nmax off-diagonal cosine similarity in the corpus: {sim[iu].max():.4f}")
print(f"pairs at threshold {THRESHOLD}:")
for a, b, s in examples:
    same = chunks[a]["text"] == chunks[b]["text"]
    print(f"  {a} '{chunks[a]['title']}' vs {b} '{chunks[b]['title']}' | cos {s} | identical text: {same}")

# ---------------------------------------------------------------------------------------------
# APPLY: final rule (decided after the report above): embedding cos >= 0.99 AND text similarity
# >= 0.98 (difflib ratio). Embedding similarity alone also catches DIFFERENT entities with similar
# wording (e.g. "Miss Nepal 2013" vs "2014"), which must not be merged.
# Then: rebuild HNSW without the dropped chunks, re-retrieve top 20 for calibration + test,
# remap gold to the kept copy, recompute labels, q_hat (calibration) and Phase 7-8 retrieval
# numbers on test. The LLM is not rerun (see the report).
# ---------------------------------------------------------------------------------------------
import difflib  # noqa: E402
import sys  # noqa: E402

import faiss  # noqa: E402

sys.path.insert(0, "src")
from conformal import conformal_qhat  # noqa: E402
from data_loader import load_raw  # noqa: E402
from index import build_index, get_model, embed_texts, search_vec  # noqa: E402

COS_MIN, TEXT_MIN = 0.99, 0.98
i, j = np.where(np.triu(sim >= COS_MIN, k=1))
keep_of = {}
for a, b in sorted(zip(i.tolist(), j.tolist())):
    if difflib.SequenceMatcher(None, chunks[a]["text"], chunks[b]["text"]).ratio() >= TEXT_MIN:
        root = keep_of.get(a, a)
        if b not in keep_of and b != root:
            keep_of[b] = root
dropped = sorted(keep_of)
kept = np.array([c for c in range(len(chunks)) if c not in keep_of])
print(f"\nAPPLY rule cos >= {COS_MIN} and text sim >= {TEXT_MIN}: drop {len(dropped)} chunks: "
      f"{[chunks[d]['title'][:30] for d in dropped]}")

index = build_index(emb[kept])
model = get_model()
raw = {r["id"]: r for r in load_raw()}
tc = np.array([c["token_cost"] for c in chunks])


def remap(c):
    return keep_of.get(c, c)


def run_split(split):
    qs = splits[split]
    I, _ = search_vec(index, embed_texts(model, [raw[q]["question"] for q in qs]), 20)
    out = []
    for q, row in zip(qs, I):
        ids = [int(kept[x]) for x in row]
        gq = sorted({remap(c) for c in gold[q]})
        cum = np.cumsum(tc[ids])
        pos = sorted(ids.index(c) for c in gq if c in ids)
        inpool = len(pos) == len(gq)
        out.append({"query_id": q, "ids": ids, "cum": cum, "in_pool": inpool,
                    "g_tokens": float(cum[pos[-1]]) if inpool else np.nan,
                    "g_rank": pos[-1] + 1 if inpool else np.nan})
    return out


cal, te = run_split("calibration"), run_split("test")
gc = np.array([r["g_tokens"] for r in cal if r["in_pool"]])
q_hat = conformal_qhat(gc, 0.1)
k_conf = int(conformal_qhat(np.array([r["g_rank"] for r in cal if r["in_pool"]]), 0.1))


def summarize(rs, budget=None, k=None):
    cov, toks = [], []
    for r in rs:
        if budget is not None:
            fits = r["cum"] <= budget
            toks.append(r["cum"][fits].max() if fits.any() else 0.0)
            cov.append(r["in_pool"] and r["g_tokens"] <= budget)
        else:
            toks.append(r["cum"][k - 1])
            cov.append(r["in_pool"] and r["g_rank"] <= k)
    cov, pool = np.array(cov), np.array([r["in_pool"] for r in rs])
    return {"recall_both": cov.mean(), "cov_conditional": cov[pool].mean(), "tokens": float(np.mean(toks)),
            "in_pool": pool.mean()}


res = {"global_conformal": summarize(te, budget=q_hat), f"fixed_k{k_conf}_cal": summarize(te, k=k_conf),
       "fixed_k10": summarize(te, k=10), "fixed_k20": summarize(te, k=20)}
d = pd.DataFrame(res).T.round(4)
d["saved_vs_k20_pct"] = (100 * (1 - d.tokens / d.loc["fixed_k20", "tokens"])).round(1)
print(f"after dedup: q_hat (calibration) = {q_hat:.0f} | conformal k from calibration = {k_conf} | "
      f"calibration coverable {np.mean([r['in_pool'] for r in cal]):.3f}")
print(d.to_string())
d.to_csv("ablations/phase9_dedup/after_dedup_test.csv")
json.dump({"rule": {"cos_min": COS_MIN, "text_min": TEXT_MIN}, "dropped": dropped,
           "kept_copy_of": {str(k): v for k, v in keep_of.items()}, "q_hat": q_hat, "k_conf": k_conf},
          open("ablations/phase9_dedup/dedup_applied.json", "w"), indent=1)
changed = sum(set(r["ids"]) != set(ret[r["query_id"]]["ids"]) for r in te)
print(f"test top-20 lists that changed after dedup: {changed} of {len(te)}")
