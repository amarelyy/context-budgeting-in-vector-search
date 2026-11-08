"""Phase 6 methods compared on dev data (NO TEST): global vs 3 bins (Mondrian) vs CQR.

All model outputs for FIT rows are OUT-OF-FOLD (5-fold): the logistic score (for bins) and
the quantile regression prediction (for CQR). Calibration rows use models fitted on all of fit.
Fit + calibration pooled (425), 500 random re-splits: 75 calibrate, 350 evaluate.
Method A (conditional coverage), alpha 0.1. Fixed-k reference on the same queries.

Usage (repo root):
  python ablations\phase6_methods\run_methods.py          # 5 distance signals
  python ablations\phase6_methods\run_methods.py --qtype  # + question-type signal (option 3)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src")
from conformal import conformal_qhat, bin_edges_from, assign_bins  # noqa: E402
from score_model import make_model  # noqa: E402
from cqr import make_qr, SIGNALS, load_table  # noqa: E402

OUT = "ablations/phase6_methods"
ALPHA, N_REPEATS, N_CAL, SEED, N_BINS = 0.1, 500, 75, 0, 3
QTYPE = "--qtype" in sys.argv
FEATS = SIGNALS + (["q_comparison"] if QTYPE else [])
TAG = "qtype" if QTYPE else "base"

df = load_table("data/processed/question_type.csv" if QTYPE else None)
meta = json.load(open("data/processed/score_model_meta.json"))
df["y"] = (df.gold_pos_tokens <= meta["T"]).astype(int)
df = df[df.split.isin(["fit", "calibration"])].reset_index(drop=True)
is_fit = (df.split == "fit").to_numpy()
pool = df.gold_in_pool.to_numpy(dtype=bool)
g = df.gold_pos_tokens.to_numpy(dtype=float)
g_inf = np.where(pool, g, np.inf)
X = df[FEATS].to_numpy()

# ---- out-of-fold predictions for fit rows; full-fit models for calibration rows ----
conf = np.zeros(len(df))
qpred = np.zeros(len(df))
fit_idx = np.where(is_fit)[0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in cv.split(fit_idx, df.y.to_numpy()[fit_idx]):
    tr, te = fit_idx[tr], fit_idx[te]
    conf[te] = make_model().fit(X[tr], df.y.to_numpy()[tr]).predict_proba(X[te])[:, 1]
    trc = tr[pool[tr]]
    qpred[te] = make_qr().fit(X[trc], np.log(g[trc])).predict(X[te])
cal_idx = np.where(~is_fit)[0]
fc = fit_idx[pool[fit_idx]]
conf[cal_idx] = make_model().fit(X[fit_idx], df.y.to_numpy()[fit_idx]).predict_proba(X[cal_idx])[:, 1]
qpred[cal_idx] = make_qr().fit(X[fc], np.log(g[fc])).predict(X[cal_idx])

from sklearn.metrics import roc_auc_score  # noqa: E402
auroc_oof = roc_auc_score(df.y.to_numpy()[fit_idx], conf[fit_idx])

chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
tc = np.array([c["token_cost"] for c in chunks])
ret = json.load(open("data/processed/retrieval_top20.json"))
cum = np.array([np.cumsum(tc[ret[q]["ids"]]) for q in df.query_id])
k20 = cum[:, -1]


def fed(budget, rows):
    c = cum[rows]
    fits = c <= budget[:, None]
    out = np.where(fits.any(axis=1), (c * fits).max(axis=1), 0.0)
    return np.where(np.isinf(budget), k20[rows], out)


rng = np.random.default_rng(SEED)
recs = []
for rep in range(N_REPEATS):
    perm = rng.permutation(len(df))
    cal, ev = perm[:N_CAL], perm[N_CAL:]
    calc = cal[pool[cal]]
    out = {}
    out["global"] = np.full(len(ev), conformal_qhat(g[calc], ALPHA))
    edges = bin_edges_from(conf[ev], N_BINS)
    bc, be = assign_bins(conf[cal], edges), assign_bins(conf[ev], edges)
    qh = [conformal_qhat(g[cal][(bc == k) & pool[cal]], ALPHA) for k in range(N_BINS)]
    out["bins3"] = np.array(qh)[be]
    Q = conformal_qhat(np.log(g[calc]) - qpred[calc], ALPHA)
    out["cqr"] = np.exp(qpred[ev] + Q)
    for m, budget in out.items():
        cov = pool[ev] & (g_inf[ev] <= budget)
        f = fed(budget, ev)
        recs.append({"repeat": rep, "method": m, "cov_conditional": cov[pool[ev]].mean(),
                     "cov_unconditional": cov.mean(), "fed_tokens": f.mean(), "k20_tokens": k20[ev].mean()})

raw = pd.DataFrame(recs)
raw["saved_pct"] = 100 * (1 - raw.fed_tokens / raw.k20_tokens)
raw.to_csv(f"{OUT}/raw_repeats_{TAG}.csv", index=False)
summ = raw.groupby("method").agg(
    cov_conditional=("cov_conditional", "mean"),
    cov_conditional_p05=("cov_conditional", lambda x: x.quantile(0.05)),
    cov_unconditional=("cov_unconditional", "mean"),
    fed_tokens=("fed_tokens", "mean"), saved_pct=("saved_pct", "mean"),
    saved_pct_std=("saved_pct", "std")).round(4).reindex(["global", "bins3", "cqr"]).reset_index()
summ.insert(0, "features", TAG)
summ.to_csv(f"{OUT}/summary_{TAG}.csv", index=False)

p = raw.pivot(index="repeat", columns="method", values="fed_tokens")
print(f"features = {FEATS}")
print(f"out-of-fold logistic AUROC on fit (target y): {auroc_oof:.3f}")
print(summ.to_string(index=False))
for m in ("bins3", "cqr"):
    d = p[m] - p["global"]
    print(f"paired vs global: {m} {d.mean():+.1f} tokens (cheaper in {(d < 0).mean():.1%} of re-splits)")
rk = np.nan_to_num(df.gold_pos_rank.to_numpy(), nan=99)
for k in (10, 11, 12):
    c = rk <= k
    print(f"fixed k={k}: cond cov {c[pool].mean():.3f} | tokens {cum[:, k - 1].mean():.0f} | "
          f"saved {100 * (1 - cum[:, k - 1].mean() / k20.mean()):.1f}%")
