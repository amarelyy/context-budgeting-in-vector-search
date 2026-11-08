"""Phase 9.2: alpha sweep + conformal vs a simple grid-search threshold.

Methods (all calibrated on coverable calibration queries, method A, one global setting):
  conformal_tokens : split-conformal token budget (finite-sample correction)  <- our main method
  grid_tokens      : smallest budget on a 25-token grid whose EMPIRICAL calibration coverage
                     >= 1 - alpha (no finite-sample correction): the "hand-tuned threshold" rival
  conformal_k      : fixed-k with k = conformal quantile of gold_pos_rank (chunks, not tokens)
Part A (dev, no test): fit + calibration pooled, 500 re-splits (75 calibrate / 350 evaluate).
Part B (test, reporting only, alpha was fixed at 0.1 before): calibrate on the real calibration
split, evaluate on test.
Outputs: summary_dev.csv, summary_test.csv, alpha_sweep.png
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from conformal import conformal_qhat  # noqa: E402

OUT = "ablations/phase9_alpha"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
GRID = np.arange(0, 6001, 25)
N_REPEATS, N_CAL = 500, 75

lab = pd.read_csv("data/processed/labels.csv")
chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
tc = np.array([c["token_cost"] for c in chunks])
ret = json.load(open("data/processed/retrieval_top20.json"))


def arrays(d):
    cum = np.array([np.cumsum(tc[ret[q]["ids"]]) for q in d.query_id])
    return (d.gold_pos_tokens.to_numpy(float), d.gold_pos_rank.to_numpy(float),
            d.gold_in_pool.to_numpy(bool), cum)


def grid_budget(scores, alpha):
    for b in GRID:
        if (scores <= b).mean() >= 1 - alpha:
            return float(b)
    return np.inf


def conformal_rank(ranks, alpha):
    return conformal_qhat(ranks, alpha)


def evaluate(budget_tokens, k, g, rk, pool, cum):
    """Returns (cond. coverage, uncond. coverage, tokens fed) for a token budget or a fixed k."""
    if budget_tokens is not None:
        cov = pool & (np.nan_to_num(g, nan=np.inf) <= budget_tokens)
        fits = cum <= budget_tokens
        fed = np.where(fits.any(axis=1), (cum * fits).max(axis=1), 0.0) if np.isfinite(budget_tokens) else cum[:, -1]
    else:
        kk = 20 if not np.isfinite(k) else int(min(k, 20))
        cov = pool & (np.nan_to_num(rk, nan=99) <= kk)
        fed = cum[:, kk - 1]
    return cov[pool].mean(), cov.mean(), fed.mean()


def run_one(cal, ev, alpha):
    gc, rc, pc, _ = cal
    g, rk, pool, cum = ev
    out = {}
    out["conformal_tokens"] = evaluate(conformal_qhat(gc[pc], alpha), None, g, rk, pool, cum)
    out["grid_tokens"] = evaluate(grid_budget(gc[pc], alpha), None, g, rk, pool, cum)
    out["conformal_k"] = evaluate(None, conformal_rank(rc[pc], alpha), g, rk, pool, cum)
    return out


# ---- Part A: dev ----
dev = lab[lab.split.isin(["fit", "calibration"])].reset_index(drop=True)
G, R, P, C = arrays(dev)
rng = np.random.default_rng(0)
recs = []
for rep in range(N_REPEATS):
    perm = rng.permutation(len(dev))
    ca, ev = perm[:N_CAL], perm[N_CAL:]
    for a in ALPHAS:
        for m, (cc, cu, tk) in run_one((G[ca], R[ca], P[ca], C[ca]), (G[ev], R[ev], P[ev], C[ev]), a).items():
            recs.append({"alpha": a, "method": m, "cov_conditional": cc, "cov_unconditional": cu, "tokens": tk,
                         "k20": C[ev][:, -1].mean()})
raw = pd.DataFrame(recs)
raw["saved_pct"] = 100 * (1 - raw.tokens / raw.k20)
raw["undercover"] = raw.cov_conditional < (1 - raw.alpha)
dev_s = raw.groupby(["method", "alpha"]).agg(
    target=("alpha", lambda x: 1 - x.iloc[0]), cov_conditional=("cov_conditional", "mean"),
    cov_p05=("cov_conditional", lambda x: x.quantile(0.05)), share_below_target=("undercover", "mean"),
    cov_unconditional=("cov_unconditional", "mean"), tokens=("tokens", "mean"),
    saved_pct=("saved_pct", "mean")).round(4).reset_index()
dev_s.to_csv(f"{OUT}/summary_dev.csv", index=False)
print("=== Part A: dev (500 re-splits) ===")
print(dev_s.to_string(index=False))

# ---- Part B: test (reporting) ----
cal = lab[lab.split == "calibration"]
te = lab[lab.split == "test"]
ca, tt = arrays(cal), arrays(te)
rows = []
for a in ALPHAS:
    for m, (cc, cu, tk) in run_one(ca, tt, a).items():
        rows.append({"alpha": a, "method": m, "target": 1 - a, "cov_conditional": cc, "cov_unconditional": cu,
                     "tokens": tk, "saved_pct": 100 * (1 - tk / tt[3][:, -1].mean())})
test_s = pd.DataFrame(rows).round(4)
test_s.to_csv(f"{OUT}/summary_test.csv", index=False)
print("\n=== Part B: test (calibrated on the real calibration split) ===")
print(test_s.to_string(index=False))

# ---- plot (dev) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for m, col in (("conformal_tokens", "#3b6fb6"), ("grid_tokens", "#d08a2e"), ("conformal_k", "#2e8b57")):
    s = dev_s[dev_s.method == m]
    ax[0].plot(s.alpha, s.cov_conditional, "o-", color=col, label=m)
    ax[1].plot(s.cov_conditional, s.tokens, "o-", color=col, label=m)
ax[0].plot(ALPHAS, [1 - a for a in ALPHAS], "k:", label="target 1 - alpha")
ax[0].set_xlabel("alpha"); ax[0].set_ylabel("conditional coverage (dev mean)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("conditional coverage (dev mean)"); ax[1].set_ylabel("tokens fed (dev mean)"); ax[1].legend(fontsize=8)
ax[1].set_title("trade-off: lower-right is better")
fig.suptitle("Phase 9.2 alpha sweep (dev: fit + calibration, 500 re-splits)")
fig.tight_layout(); fig.savefig(f"{OUT}/alpha_sweep.png", dpi=110)
print("saved", OUT)
