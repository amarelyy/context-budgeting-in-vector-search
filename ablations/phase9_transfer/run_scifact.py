"""Phase 9.5 (stretch): transfer to a BEIR dataset (SciFact, starting choice), retrieval only.

Same pipeline as HotpotQA: one chunk = one document's text (abstract), Qwen3 token_cost,
all-MiniLM-L6-v2 embeddings (L2-normalized), FAISS HNSW (same parameters), top 20.
Queries: SciFact test queries with qrels. Covered = ALL relevant docs inside the fed set
(the analogue of "both gold"). No LLM: SciFact has no short answers for EM/F1.

A. No re-calibration: apply the HotpotQA settings as they are
   - global token budget q_hat = 1649 (data/processed/conformal.json)
   - fixed k = 12 (conformal on ranks, HotpotQA calibration)
   - CQR (5 signals, model + Q from HotpotQA)
B. Re-calibrated on SciFact: 500 re-splits (half calibrate / half evaluate), alpha 0.1, method A,
   global token budget, conformal k, and CQR with the HotpotQA quantile model but a SciFact Q.
Outputs in ablations/phase9_transfer/.
"""
import json
import os
import sys

import faiss
import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, "src")
from conformal import conformal_qhat  # noqa: E402
from index import build_index, get_model, embed_texts, search_vec  # noqa: E402
from signals import compute_signals  # noqa: E402
from cqr import SIGNALS  # noqa: E402

OUT = "ablations/phase9_transfer"
ALPHA, N_REPEATS = 0.1, 500

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qrels = load_dataset("BeIR/scifact-qrels", split="test")

doc_ids = [str(x) for x in corpus["_id"]]
texts = [t.strip() for t in corpus["text"]]
pos_of = {d: i for i, d in enumerate(doc_ids)}
rel = {}
for r in qrels:
    if r["score"] > 0:
        rel.setdefault(str(r["query-id"]), set()).add(pos_of[str(r["corpus-id"])])
qtext = {str(q["_id"]): q["text"] for q in queries}
qids = sorted(rel, key=int)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tc = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])
model = get_model()
emb = embed_texts(model, texts)
index = build_index(emb)
I, D = search_vec(index, embed_texts(model, [qtext[q] for q in qids]), 20)

rows = []
for q, ids, d in zip(qids, I, D):
    ids = ids.tolist()
    cum = np.cumsum(tc[ids])
    pos = sorted(ids.index(c) for c in rel[q] if c in ids)
    inpool = len(pos) == len(rel[q])
    rows.append({"query_id": q, "n_relevant": len(rel[q]), "in_pool": inpool,
                 "g_tokens": float(cum[pos[-1]]) if inpool else np.nan,
                 "g_rank": pos[-1] + 1 if inpool else np.nan, "cum": cum, **compute_signals(d)})
df = pd.DataFrame(rows)
cum = np.stack(df.cum.to_numpy())
pool = df.in_pool.to_numpy(bool)
g = df.g_tokens.to_numpy(float)
rk = df.g_rank.to_numpy(float)
k20 = cum[:, -1]

print(f"SciFact: {len(texts)} docs | token_cost median {np.median(tc):.0f} (HotpotQA: 117.5), "
      f"mean {tc.mean():.0f}, max {tc.max()} | docs > 256 tokens: {(tc > 256).mean():.2f}")
print(f"test queries {len(df)} | relevant docs per query: {df.n_relevant.value_counts().sort_index().to_dict()}")
print(f"all relevant in top 20: {pool.mean():.3f} | k=20 tokens mean {k20.mean():.0f}")


def ev(budget=None, k=None, rows_=slice(None)):
    c, p_, gg, rr = cum[rows_], pool[rows_], g[rows_], rk[rows_]
    if budget is not None:
        b = np.broadcast_to(budget, (len(c),))
        cov = p_ & (np.nan_to_num(gg, nan=np.inf) <= b)
        fits = c <= b[:, None]
        fed = np.where(fits.any(axis=1), (c * fits).max(axis=1), 0.0)
    else:
        cov = p_ & (np.nan_to_num(rr, nan=99) <= k)
        fed = c[:, k - 1]
    return {"cov_conditional": cov[p_].mean(), "cov_unconditional": cov.mean(), "tokens": fed.mean(),
            "avg_chunks": float((c <= (np.broadcast_to(budget, (len(c),))[:, None] if budget is not None
                                       else c[:, k - 1:k])).sum(axis=1).mean()),
            "saved_vs_k20_pct": 100 * (1 - fed.mean() / k20[rows_].mean())}


# ---- A: no re-calibration ----
hp = json.load(open("data/processed/conformal.json"))
cqr_meta = json.load(open("data/processed/cqr.json"))
qr = joblib.load("data/processed/cqr_model.joblib")
qpred = qr.predict(df[SIGNALS])
A = {"hotpot_global_tokens (q_hat 1649)": ev(budget=hp["q_hat_global"]),
     "hotpot_fixed_k12": ev(k=12),
     "hotpot_cqr": ev(budget=np.exp(qpred + cqr_meta["Q_log"])),
     "fixed_k5": ev(k=5), "fixed_k20": ev(k=20)}
a_df = pd.DataFrame(A).T.round(4)
a_df.to_csv(f"{OUT}/transfer_no_recalibration.csv")
print("\n=== A: HotpotQA settings, no re-calibration (target conditional coverage 0.90) ===")
print(a_df.to_string())

# ---- B: re-calibrated on SciFact (500 re-splits, half / half) ----
rng = np.random.default_rng(0)
n = len(df)
recs = []
for rep in range(N_REPEATS):
    perm = rng.permutation(n)
    ca, e = perm[: n // 2], perm[n // 2:]
    cc = ca[pool[ca]]
    qh = conformal_qhat(g[cc], ALPHA)
    kc = int(min(conformal_qhat(rk[cc], ALPHA), 20))
    Qs = conformal_qhat(np.log(g[cc]) - qpred[cc], ALPHA)
    for m, kw in (("recal_global_tokens", {"budget": qh}), ("recal_conformal_k", {"k": kc}),
                  ("recal_cqr", {"budget": np.exp(qpred[e] + Qs)})):
        r = ev(rows_=e, **kw)
        r.update({"method": m, "repeat": rep, "q_hat": qh, "k": kc})
        recs.append(r)
b = pd.DataFrame(recs)
b_s = b.groupby("method").agg(cov_conditional=("cov_conditional", "mean"),
                              cov_p05=("cov_conditional", lambda x: x.quantile(0.05)),
                              cov_unconditional=("cov_unconditional", "mean"),
                              tokens=("tokens", "mean"), saved_vs_k20_pct=("saved_vs_k20_pct", "mean"),
                              q_hat_median=("q_hat", "median"), k_median=("k", "median")).round(4)
b_s.to_csv(f"{OUT}/transfer_recalibrated.csv")
print(f"\n=== B: re-calibrated on SciFact ({N_REPEATS} re-splits, {n // 2} calibrate / {n - n // 2} evaluate) ===")
print(b_s.to_string())
df.drop(columns="cum").to_csv(f"{OUT}/scifact_queries.csv", index=False)
