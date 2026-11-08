"""Phase 6 dev estimate: does a per-bin (Mondrian) budget beat one global budget? NO TEST DATA.

- conf_score for FIT queries is OUT-OF-FOLD (5-fold cross_val_predict), so no query is scored
  by a model that saw it. Calibration queries use the model fitted on all of fit (also unseen).
- Fit + calibration pooled (425), 500 random re-splits: 75 to calibrate q_hat, 350 to evaluate.
- Bin edges = equal-count quantiles of conf_score on the 350 evaluation-side queries' complement
  (i.e. NOT the 75 calibration queries), so edges and q_hat use different data.
- Method A (conditional), alpha 0.1. n_bins = 1 (global), 2, 3.
- Reference: fixed-k on the same 425 queries.
Run from the repo root.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, "src")
from conformal import conformal_qhat, bin_edges_from, assign_bins  # noqa: E402
from score_model import make_model, SIGNALS  # noqa: E402

OUT = "ablations/phase6_mondrian"
ALPHA, N_REPEATS, N_CAL, SEED = 0.1, 500, 75, 0

s = pd.read_csv("data/processed/signals.csv")
lab = pd.read_csv("data/processed/labels.csv")
sc = pd.read_csv("data/processed/conf_scores.csv")
df = s.merge(lab, on=["query_id", "split"]).merge(sc, on=["query_id", "split"])
df = df[df.split.isin(["fit", "calibration"])].reset_index(drop=True)

# out-of-fold conf_score for fit queries
fit_mask = (df.split == "fit").to_numpy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(make_model(), df.loc[fit_mask, SIGNALS], df.loc[fit_mask, "y"],
                        cv=cv, method="predict_proba")[:, 1]
conf = df.conf_score.to_numpy().copy()
conf[fit_mask] = oof

chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
tc = np.array([c["token_cost"] for c in chunks])
ret = json.load(open("data/processed/retrieval_top20.json"))
cum = np.array([np.cumsum(tc[ret[q]["ids"]]) for q in df.query_id])
g = df.gold_pos_tokens.to_numpy(dtype=float)
pool = df.gold_in_pool.to_numpy(dtype=bool)
g_inf = np.where(pool, g, np.inf)
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
    for n_bins in (1, 2, 3):
        if n_bins == 1:
            budget = np.full(len(ev), conformal_qhat(g[cal][pool[cal]], ALPHA))
            n_min = int(pool[cal].sum())
        else:
            edges = bin_edges_from(conf[ev], n_bins)          # edges NOT from calibration rows
            bc, be = assign_bins(conf[cal], edges), assign_bins(conf[ev], edges)
            qh = [conformal_qhat(g[cal][(bc == k) & pool[cal]], ALPHA) for k in range(n_bins)]
            budget = np.array(qh)[be]
            n_min = min(int(((bc == k) & pool[cal]).sum()) for k in range(n_bins))
        cov = pool[ev] & (g_inf[ev] <= budget)
        f = fed(budget, ev)
        recs.append({"repeat": rep, "n_bins": n_bins, "cov_conditional": cov[pool[ev]].mean(),
                     "cov_unconditional": cov.mean(), "fed_tokens": f.mean(),
                     "k20_tokens": k20[ev].mean(), "any_inf": bool(np.isinf(budget).any()),
                     "min_coverable_per_bin": n_min})

raw = pd.DataFrame(recs)
raw["saved_pct"] = 100 * (1 - raw.fed_tokens / raw.k20_tokens)
raw.to_csv(f"{OUT}/raw_repeats.csv", index=False)
summ = raw.groupby("n_bins").agg(
    cov_conditional=("cov_conditional", "mean"),
    cov_conditional_p05=("cov_conditional", lambda x: x.quantile(0.05)),
    cov_unconditional=("cov_unconditional", "mean"),
    fed_tokens=("fed_tokens", "mean"), saved_pct=("saved_pct", "mean"),
    saved_pct_std=("saved_pct", "std"),
    share_with_inf_bin=("any_inf", "mean"),
    min_coverable_per_bin_median=("min_coverable_per_bin", "median"),
).round(4).reset_index()
summ.to_csv(f"{OUT}/summary.csv", index=False)

# paired difference: 3 bins vs global, same re-split
p = raw.pivot(index="repeat", columns="n_bins", values="fed_tokens")
d3, d2 = p[3] - p[1], p[2] - p[1]
print(f"pooled {len(df)} queries | {N_REPEATS} re-splits | alpha {ALPHA} | method A | "
      f"out-of-fold conf_score for fit")
print(summ.to_string(index=False))
print(f"\npaired fed-token difference vs global: 2 bins {d2.mean():+.1f} (bins cheaper in "
      f"{(d2 < 0).mean():.2%} of re-splits) | 3 bins {d3.mean():+.1f} (cheaper in {(d3 < 0).mean():.2%})")

# fixed-k reference at similar conditional coverage
rk = np.nan_to_num(df.gold_pos_rank.to_numpy(), nan=99)
for k in (10, 11, 12):
    c = rk <= k
    print(f"fixed k={k}: cond cov {c[pool].mean():.3f} | uncond {c.mean():.3f} | tokens {cum[:, k - 1].mean():.0f} | "
          f"saved {100 * (1 - cum[:, k - 1].mean() / k20.mean()):.1f}%")
