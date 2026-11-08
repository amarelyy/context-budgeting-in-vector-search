"""Phase 6 ablation: how to handle queries whose gold can never be covered (24%), and which alpha.

NO TEST DATA. Fit + calibration are pooled (425 queries) and re-split at random
N_REPEATS times into a calibration part (75 queries, same size as the real split)
and an evaluation part (the other 350).

Methods (marginal split conformal, ONE global budget for all queries, no signals yet):
  A  conditional : q_hat from coverable calibration queries only.
                   Target: P(gold covered | gold in top 20) >= 1 - alpha.
  C  infinite    : uncoverable queries count as gold_pos_tokens = +inf, q_hat from all.
                   Target: P(gold covered) >= 1 - alpha. If q_hat = inf -> feed all 20.
  (B = C with a larger alpha; the alpha sweep covers it.)

q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score (finite-sample correction);
if that index > n, q_hat = inf.
Fed set = longest prefix of the ranked top 20 whose total token_cost <= q_hat.
Covered  = gold_pos_tokens <= q_hat.

Run from the repo root:  .\.venv\Scripts\python.exe ablations\phase6_coverage\run_ablation.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "ablations/phase6_coverage"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_REPEATS = 500          # random re-splits (starting value)
N_CAL = 75               # same size as the real calibration split
SEED = 0

labels = pd.read_csv("data/processed/labels.csv")
lab = labels[labels.split.isin(["fit", "calibration"])].reset_index(drop=True)
chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
tc = np.array([c["token_cost"] for c in chunks])
ret = json.load(open("data/processed/retrieval_top20.json"))
cum = np.array([np.cumsum(tc[ret[q]["ids"]]) for q in lab.query_id])   # (425, 20)

score = lab.gold_pos_tokens.to_numpy(dtype=float)        # NaN = uncoverable
in_pool = lab.gold_in_pool.to_numpy(dtype=bool)
score_inf = np.where(in_pool, score, np.inf)
pool_tokens = cum[:, -1]


def conformal_qhat(scores, alpha):
    s = np.sort(scores)
    n = len(s)
    idx = math.ceil((n + 1) * (1 - alpha))
    return np.inf if idx > n else s[idx - 1]


def fed_tokens(qhat, rows):
    """Tokens actually fed: longest prefix with cum <= qhat (all 20 if qhat = inf)."""
    if np.isinf(qhat):
        return pool_tokens[rows]
    c = cum[rows]
    fits = c <= qhat
    return np.where(fits.any(axis=1), (c * fits).max(axis=1), 0)


rng = np.random.default_rng(SEED)
rows_out = []
for rep in range(N_REPEATS):
    perm = rng.permutation(len(lab))
    cal, ev = perm[:N_CAL], perm[N_CAL:]
    for alpha in ALPHAS:
        for method in ("A_conditional", "C_infinite"):
            if method == "A_conditional":
                qhat = conformal_qhat(score[cal][in_pool[cal]], alpha)
            else:
                qhat = conformal_qhat(score_inf[cal], alpha)
            # uncoverable queries are NEVER covered (inf <= inf would be True in Python)
            covered = in_pool[ev] & (score_inf[ev] <= qhat)
            fed = fed_tokens(qhat, ev)
            rows_out.append({
                "repeat": rep, "alpha": alpha, "method": method,
                "q_hat": qhat, "q_hat_is_inf": bool(np.isinf(qhat)),
                "cov_conditional": covered[in_pool[ev]].mean(),
                "cov_unconditional": covered.mean(),
                "fed_tokens_mean": fed.mean(),
                "k20_tokens_mean": pool_tokens[ev].mean(),
            })

raw = pd.DataFrame(rows_out)
raw["tokens_saved_pct"] = 100 * (1 - raw.fed_tokens_mean / raw.k20_tokens_mean)
raw.to_csv(f"{OUT}/raw_repeats.csv", index=False)

summary = raw.groupby(["method", "alpha"]).agg(
    target=("alpha", lambda a: 1 - a.iloc[0]),
    q_hat_median=("q_hat", "median"),
    q_hat_inf_share=("q_hat_is_inf", "mean"),
    cov_conditional_mean=("cov_conditional", "mean"),
    cov_conditional_p05=("cov_conditional", lambda x: x.quantile(0.05)),
    cov_unconditional_mean=("cov_unconditional", "mean"),
    fed_tokens_mean=("fed_tokens_mean", "mean"),
    tokens_saved_pct_mean=("tokens_saved_pct", "mean"),
).reset_index().round(4)
summary.to_csv(f"{OUT}/summary.csv", index=False)

print(f"pooled fit+calibration: {len(lab)} queries, coverable {in_pool.sum()} "
      f"({in_pool.mean():.3f}) | repeats {N_REPEATS} | cal size {N_CAL}")
print(summary.to_string(index=False))

# ---- plot: coverage and tokens saved vs alpha ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for method, color in (("A_conditional", "#3b6fb6"), ("C_infinite", "#d08a2e")):
    s = summary[summary.method == method]
    axes[0].plot(s.alpha, s.cov_conditional_mean, "o-", color=color, label=f"{method}: conditional cov.")
    axes[0].plot(s.alpha, s.cov_unconditional_mean, "s--", color=color, label=f"{method}: unconditional cov.")
    axes[1].plot(s.alpha, s.tokens_saved_pct_mean, "o-", color=color, label=method)
axes[0].plot(ALPHAS, [1 - a for a in ALPHAS], "k:", label="target 1 - alpha")
axes[0].axhline(in_pool.mean(), color="gray", lw=0.8, label=f"ceiling {in_pool.mean():.3f}")
axes[0].set_xlabel("alpha"); axes[0].set_ylabel("coverage"); axes[0].legend(fontsize=7)
axes[1].set_xlabel("alpha"); axes[1].set_ylabel("tokens saved vs k=20 (%)"); axes[1].legend(fontsize=8)
fig.suptitle(f"Phase 6 ablation (fit+calibration, {N_REPEATS} random re-splits, no test data)")
fig.tight_layout()
fig.savefig(f"{OUT}/coverage_tokens_vs_alpha.png", dpi=110)
print("saved summary.csv, raw_repeats.csv, coverage_tokens_vs_alpha.png in", OUT)
