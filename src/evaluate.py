"""Phase 8 (8.1-8.4): the four checks on the TEST split.

8.1 auroc: conf_score vs y (both gold within T tokens), and the top1_dist-only rival. GATE.
8.2 ece + reliability diagram for conf_score.
8.3 coverage of the main method (global conformal budget) vs 1 - alpha.
8.4 tokens_saved_pct at equal recall_at_budget vs fixed-k (interpolated fixed-k curve),
    plus answer_f1 vs fixed-k (k = 11, 12 LLM runs added here, main setting).
Bootstrap 95% CIs (2,000 resamples of the 75 test queries, seed 0) because n is small.
Outputs: results/phase8_checks.json, results/plots/reliability_test.png,
         results/llm_fixed_k_11_12_test_main_seed42.csv
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data_loader import load_raw  # noqa: E402
from llm import chat_checked, MAIN_OPTIONS  # noqa: E402
from prompt import build_messages  # noqa: E402
from qa_metrics import exact_match, f1_score  # noqa: E402

ALPHA = 0.1
N_BOOT, SEED = 2000, 0
ECE_BINS = 5          # starting value (75 queries -> ~15 per bin)
rng = np.random.default_rng(SEED)


def boot_ci(fn, n):
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        try:
            vals.append(fn(i))
        except ValueError:      # e.g. a resample with one class only
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ece(p, y, n_bins=ECE_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out, rows = 0.0, []
    for k in range(n_bins):
        m = b == k
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((k, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out, rows


if __name__ == "__main__":
    res = {}
    sc = pd.read_csv("data/processed/conf_scores.csv")
    te = sc[sc.split == "test"].reset_index(drop=True)
    y, p, p1 = te.y.to_numpy(), te.conf_score.to_numpy(), te.conf_score_top1.to_numpy()
    n = len(te)

    # ---- 8.1 auroc ----
    a_main, a_riv = roc_auc_score(y, p), roc_auc_score(y, p1)
    res["8.1"] = {
        "n": n, "positives": int(y.sum()),
        "auroc_main": a_main, "auroc_main_ci": boot_ci(lambda i: roc_auc_score(y[i], p[i]), n),
        "auroc_top1_rival": a_riv, "auroc_top1_ci": boot_ci(lambda i: roc_auc_score(y[i], p1[i]), n),
        "diff_ci": boot_ci(lambda i: roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], p1[i]), n),
    }
    r = res["8.1"]
    print(f"8.1 AUROC (test n={n}, positives {r['positives']}): main {a_main:.3f} "
          f"[{r['auroc_main_ci'][0]:.3f}, {r['auroc_main_ci'][1]:.3f}] | rival top1_dist {a_riv:.3f} "
          f"[{r['auroc_top1_ci'][0]:.3f}, {r['auroc_top1_ci'][1]:.3f}] | diff CI "
          f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")

    # ---- 8.2 ece + reliability diagram ----
    e, rows_e = ece(p, y)
    res["8.2"] = {"ece": e, "ece_ci": boot_ci(lambda i: ece(p[i], y[i])[0], n), "bins": rows_e,
                  "mean_conf": float(p.mean()), "positive_rate": float(y.mean())}
    print(f"8.2 ECE ({ECE_BINS} bins): {e:.3f} [{res['8.2']['ece_ci'][0]:.3f}, {res['8.2']['ece_ci'][1]:.3f}] | "
          f"mean conf_score {p.mean():.3f} vs positive rate {y.mean():.3f}")
    for k, cnt, mp, my in rows_e:
        print(f"    bin {k}: n={cnt:>2} | mean conf {mp:.3f} | observed {my:.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", label="perfect calibration")
    ax.plot([r_[2] for r_ in rows_e], [r_[3] for r_ in rows_e], "o-", color="#3b6fb6", label="conf_score (test)")
    for k, cnt, mp, my in rows_e:
        ax.annotate(f"n={cnt}", (mp, my), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("mean predicted conf_score"); ax.set_ylabel("observed share easy (y = 1)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(f"Reliability diagram, test (ECE {e:.3f})")
    fig.tight_layout(); fig.savefig("results/plots/reliability_test.png", dpi=110)

    # ---- 8.3 coverage of the main method ----
    b = pd.read_csv("results/budget_test_main_seed42.csv")
    g = b[b.method == "global_conformal"].reset_index(drop=True)
    pool, cov = g.gold_in_pool.to_numpy(), g.gold_covered_both.to_numpy()
    cc = cov[pool]
    res["8.3"] = {"target": 1 - ALPHA, "cov_conditional": float(cc.mean()), "n_coverable": int(pool.sum()),
                  "cov_conditional_ci": boot_ci(lambda i: cc[i].mean(), len(cc)),
                  "cov_unconditional": float(cov.mean()), "ceiling_unconditional": float(pool.mean())}
    r = res["8.3"]
    print(f"8.3 coverage (global budget): conditional {r['cov_conditional']:.3f} "
          f"[{r['cov_conditional_ci'][0]:.3f}, {r['cov_conditional_ci'][1]:.3f}] vs target {1 - ALPHA:.2f} "
          f"(n coverable {r['n_coverable']}) | unconditional {r['cov_unconditional']:.3f} "
          f"(ceiling {r['ceiling_unconditional']:.3f})")

    # ---- 8.4 tokens saved at equal recall + LLM for fixed k = 11, 12 ----
    curve = pd.read_csv("results/fixed_k_curve_test.csv")
    out84 = {}
    for m in ("global_conformal", "bins3", "cqr"):
        x = b[b.method == m]
        rec, tok = x.gold_covered_both.mean(), x.fed_tokens.mean()
        # fixed-k tokens needed for the same recall: linear interpolation on the (recall, tokens) curve
        c = curve.drop_duplicates("recall_both", keep="first").sort_values("recall_both")
        fk_tok = float(np.interp(rec, c.recall_both, c.tokens_mean))
        out84[m] = {"recall": rec, "tokens": tok, "fixed_k_tokens_same_recall": fk_tok,
                    "tokens_saved_pct_vs_fixed_k": 100 * (1 - tok / fk_tok),
                    "tokens_saved_pct_vs_k20": 100 * (1 - tok / curve.tokens_mean.iloc[-1])}
        print(f"8.4 {m:<16}: recall {rec:.3f} | tokens {tok:.1f} | fixed-k needs {fk_tok:.1f} -> "
              f"tokens_saved_pct {out84[m]['tokens_saved_pct_vs_fixed_k']:+.1f}% "
              f"(vs k=20: {out84[m]['tokens_saved_pct_vs_k20']:.1f}%)")

    chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
    ret = json.load(open("data/processed/retrieval_top20.json"))
    gold = json.load(open("data/processed/gold_chunk_ids.json"))
    raw = {r_["id"]: r_ for r_ in load_raw()}
    recs = []
    for k in (11, 12):
        for q in te.query_id:
            fed = ret[q]["ids"][:k]
            resp, info = chat_checked(build_messages(raw[q]["question"], [chunks[c]["text"] for c in fed]),
                                      MAIN_OPTIONS)
            ans = resp["message"]["content"].strip()
            recs.append({"query_id": q, "k": k, "truncated": info["truncated"],
                         "gold_covered_both": set(gold[q]) <= set(fed), "answer": ans,
                         "gold_answer": raw[q]["answer"], "em": exact_match(ans, raw[q]["answer"]),
                         "f1": round(f1_score(ans, raw[q]["answer"]), 4)})
    with open("results/llm_fixed_k_11_12_test_main_seed42.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    fk = pd.DataFrame(recs)
    base = pd.read_csv("results/llm_baseline_test_main_seed42.csv")
    f1s = {f"fixed_k{k}": float(base[base.k == k].f1.mean()) for k in (5, 10, 20)}
    f1s.update({f"fixed_k{k}": float(fk[fk.k == k].f1.mean()) for k in (11, 12)})
    f1s.update({m: float(b[b.method == m].f1.mean()) for m in ("global_conformal", "bins3", "cqr")})
    gq = g.set_index("query_id").f1
    k12 = fk[fk.k == 12].set_index("query_id").f1.reindex(gq.index)
    dd = (gq - k12).to_numpy()
    res["8.4"] = {"tokens": out84, "answer_f1": f1s, "f1_global_minus_k12": float(dd.mean()),
                  "f1_global_minus_k12_ci": boot_ci(lambda i: dd[i].mean(), len(dd)),
                  "truncated_k11_k12": int(fk.truncated.sum())}
    print("8.4 answer_f1 (main setting):", {k_: round(v, 3) for k_, v in f1s.items()})
    print(f"    F1 global - fixed k=12 (paired): {dd.mean():+.3f} "
          f"[{res['8.4']['f1_global_minus_k12_ci'][0]:+.3f}, {res['8.4']['f1_global_minus_k12_ci'][1]:+.3f}] | "
          f"truncated {res['8.4']['truncated_k11_k12']}")

    with open("results/phase8_checks.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/phase8_checks.json, results/plots/reliability_test.png")
