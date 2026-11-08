"""Phase 6 (6.1-6.3): conformal token budget q_hat, one per conf_score bin (Mondrian).

Method A (conditional coverage, decided 2026-11-08): q_hat uses COVERABLE calibration
queries only. Guarantee (on average): if both gold are in the top 20, they are inside
the budget with probability >= 1 - alpha.

q_hat = the ceil((n+1)(1-alpha))-th smallest gold_pos_tokens in the bin (finite-sample
correction). If that index > n (bin too small), q_hat = inf -> feed all 20 chunks.
Bin edges = tertiles of conf_score on the FIT split (not chosen on calibration).
"""
import json
import math

import numpy as np
import pandas as pd

ALPHA = 0.1      # decided 2026-11-08
N_BINS = 3       # starting value
OUT = "data/processed/conformal.json"


def conformal_qhat(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return np.inf
    idx = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if idx > n else float(s[idx - 1])


def bin_edges_from(scores, n_bins=N_BINS):
    """Inner edges (n_bins - 1 values) at equal-count quantiles."""
    return [float(np.quantile(scores, q)) for q in np.linspace(0, 1, n_bins + 1)[1:-1]]


def assign_bins(conf, edges):
    return np.searchsorted(edges, conf, side="right")   # 0 = lowest confidence


def fit_mondrian(conf, gold_pos_tokens, in_pool, edges, alpha=ALPHA):
    b = assign_bins(conf, edges)
    qhats, counts = [], []
    for k in range(len(edges) + 1):
        m = (b == k) & in_pool
        qhats.append(conformal_qhat(gold_pos_tokens[m], alpha))
        counts.append(int(m.sum()))
    return qhats, counts


def budget_for(conf, edges, qhats):
    return np.array(qhats)[assign_bins(conf, edges)]


if __name__ == "__main__":
    sc = pd.read_csv("data/processed/conf_scores.csv")
    lab = pd.read_csv("data/processed/labels.csv")
    df = sc.merge(lab, on=["query_id", "split"])
    fit, cal = df[df.split == "fit"], df[df.split == "calibration"]

    edges = bin_edges_from(fit.conf_score.to_numpy())
    g = cal.gold_pos_tokens.to_numpy(dtype=float)
    pool = cal.gold_in_pool.to_numpy(dtype=bool)
    conf = cal.conf_score.to_numpy()

    # 6.1 global q_hat (no bins) and 6.2 one q_hat per bin
    q_global = conformal_qhat(g[pool])
    qhats, counts = fit_mondrian(conf, g, pool, edges)
    b = assign_bins(conf, edges)
    print(f"alpha = {ALPHA} | method A (coverable only) | calibration n = {len(cal)}, coverable {pool.sum()}")
    print(f"bin edges (fit conf_score tertiles): {[round(e, 4) for e in edges]}")
    print(f"global q_hat = {q_global:.0f} tokens")
    for k in range(N_BINS):
        m = b == k
        print(f"  bin {k} ({['low', 'mid', 'high'][k]} conf): calibration n = {m.sum()}, "
              f"coverable n = {counts[k]}, median gold_pos_tokens = "
              f"{np.nanmedian(g[m & pool]) if (m & pool).any() else float('nan'):.0f}, q_hat = {qhats[k]:.0f}")

    # 6.3 sanity check: coverage ON calibration itself (in-sample, optimistic by design)
    budget = budget_for(conf, edges, qhats)
    cov_c = (g[pool] <= budget[pool]).mean()
    cov_u = (pool & (np.nan_to_num(g, nan=np.inf) <= budget)).mean()
    cov_c_g = (g[pool] <= q_global).mean()
    print(f"\n6.3 coverage on calibration (in-sample): Mondrian conditional {cov_c:.3f} | "
          f"unconditional {cov_u:.3f} | global conditional {cov_c_g:.3f} | target {1 - ALPHA:.2f}")
    print(f"mean budget on calibration: Mondrian {budget.mean():.0f} | global {q_global:.0f}")

    with open(OUT, "w") as f:
        json.dump({"alpha": ALPHA, "method": "A_conditional", "n_bins": N_BINS, "edges": edges,
                   "q_hat_bins": qhats, "coverable_per_bin": counts, "q_hat_global": q_global}, f, indent=1)
    print("saved", OUT)
