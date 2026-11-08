"""Phase 9.1: signal ablation (dev data only: fit + calibration, NO TEST).

For each feature set: remove one signal at a time from the 5 distance signals, plus extra sets
(top1_dist only, + question type, question type only). Measure:
  - score AUROC: out-of-fold on fit (target y = both gold within T = 349), mean over 5 CV seeds
  - bins3 and CQR budgets: conditional coverage and tokens fed over 500 re-splits (75 / 350),
    method A, alpha 0.1, same protocol as ablations/phase6_methods
Global budget uses no signals, so it is the same for every set (reference row).
Output: ablations/phase9_signals/summary.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src")
from conformal import conformal_qhat, bin_edges_from, assign_bins  # noqa: E402
from score_model import make_model  # noqa: E402
from cqr import make_qr, SIGNALS, load_table  # noqa: E402

OUT = "ablations/phase9_signals"
ALPHA, N_REPEATS, N_CAL, N_BINS = 0.1, 500, 75, 3

df = load_table("data/processed/question_type.csv")
T = json.load(open("data/processed/score_model_meta.json"))["T"]
df["y"] = (df.gold_pos_tokens <= T).astype(int)
df = df[df.split.isin(["fit", "calibration"])].reset_index(drop=True)
is_fit = (df.split == "fit").to_numpy()
pool = df.gold_in_pool.to_numpy(bool)
g = df.gold_pos_tokens.to_numpy(float)
g_inf = np.where(pool, g, np.inf)
yv = df.y.to_numpy()
fit_idx, cal_idx = np.where(is_fit)[0], np.where(~is_fit)[0]

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


def oof(feats, seed=42):
    X = df[feats].to_numpy()
    conf, qp = np.zeros(len(df)), np.zeros(len(df))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(fit_idx, yv[fit_idx]):
        tr, te = fit_idx[tr], fit_idx[te]
        conf[te] = make_model().fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]
        trc = tr[pool[tr]]
        qp[te] = make_qr().fit(X[trc], np.log(g[trc])).predict(X[te])
    fc = fit_idx[pool[fit_idx]]
    conf[cal_idx] = make_model().fit(X[fit_idx], yv[fit_idx]).predict_proba(X[cal_idx])[:, 1]
    qp[cal_idx] = make_qr().fit(X[fc], np.log(g[fc])).predict(X[cal_idx])
    return conf, qp


sets = {"all 5": SIGNALS}
sets.update({f"- {s}": [x for x in SIGNALS if x != s] for s in SIGNALS})
sets.update({"top1_dist only": ["top1_dist"], "all 5 + q_comparison": SIGNALS + ["q_comparison"],
             "q_comparison only": ["q_comparison"]})

rows = []
for name, feats in sets.items():
    aucs = [roc_auc_score(yv[fit_idx], oof(feats, s)[0][fit_idx]) for s in range(5)]
    conf, qp = oof(feats)
    rng = np.random.default_rng(0)
    acc = {"bins3": [], "cqr": [], "global": []}
    for _ in range(N_REPEATS):
        perm = rng.permutation(len(df))
        cal, ev = perm[:N_CAL], perm[N_CAL:]
        calc = cal[pool[cal]]
        edges = bin_edges_from(conf[ev], N_BINS)
        bc, be = assign_bins(conf[cal], edges), assign_bins(conf[ev], edges)
        qh = [conformal_qhat(g[cal][(bc == k) & pool[cal]], ALPHA) for k in range(N_BINS)]
        Q = conformal_qhat(np.log(g[calc]) - qp[calc], ALPHA)
        for m, budget in (("bins3", np.array(qh)[be]), ("cqr", np.exp(qp[ev] + Q)),
                          ("global", np.full(len(ev), conformal_qhat(g[calc], ALPHA)))):
            cov = pool[ev] & (g_inf[ev] <= budget)
            acc[m].append((cov[pool[ev]].mean(), fed(budget, ev).mean()))
    r = {"features": name, "n_features": len(feats),
         "auroc_oof_fit": round(float(np.mean(aucs)), 4), "auroc_std": round(float(np.std(aucs)), 4)}
    for m in ("bins3", "cqr"):
        a = np.array(acc[m])
        r[f"{m}_cov"] = round(a[:, 0].mean(), 4)
        r[f"{m}_tokens"] = round(a[:, 1].mean(), 1)
    r["global_tokens"] = round(np.array(acc["global"])[:, 1].mean(), 1)
    rows.append(r)
    print(r)

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/summary.csv", index=False)
print("\n" + out.to_string(index=False))
