"""Phase 7 (7.1-7.3): token-budget retrieval on the TEST split.

Pre-declared (2026-11-08, before any test run):
  MAIN method   = global conformal budget (q_hat_global from data/processed/conformal.json)
  secondary     = 3 bins (Mondrian) and CQR (5 signals); reported, never chosen from on test
Budget rule: add chunks in ranked order (top 20) while the running total of token_cost <= budget.
LLM: main setting (temperature 0, seed 42), same prompt and num_ctx as the fixed-k baseline.

Outputs: results/budget_test_main_seed42.csv (one row per query per method),
         results/fixed_k_curve_test.csv (fixed-k k=1..20 retrieval curve on test)
"""
import csv
import json
import time

import joblib
import numpy as np
import pandas as pd

from conformal import assign_bins
from cqr import SIGNALS
from data_loader import load_raw
from llm import chat_checked, MAIN_OPTIONS
from prompt import build_messages
from qa_metrics import exact_match, f1_score

SPLIT = "test"
OUT = "results/budget_test_main_seed42.csv"


def prefix_within(ids, token_cost, budget):
    fed, total = [], 0
    for c in ids:
        if total + token_cost[c] > budget:
            break
        fed.append(c)
        total += token_cost[c]
    return fed


if __name__ == "__main__":
    chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
    tc = np.array([c["token_cost"] for c in chunks])
    ret = json.load(open("data/processed/retrieval_top20.json"))
    gold = json.load(open("data/processed/gold_chunk_ids.json"))
    rows = {r["id"]: r for r in load_raw()}
    conf = json.load(open("data/processed/conformal.json"))
    cqr = json.load(open("data/processed/cqr.json"))
    qr = joblib.load("data/processed/cqr_model.joblib")
    sig = pd.read_csv("data/processed/signals.csv")
    sc = pd.read_csv("data/processed/conf_scores.csv")
    t = sig.merge(sc, on=["query_id", "split"])
    t = t[t.split == SPLIT].set_index("query_id")
    qids = json.load(open("data/processed/splits.json"))[SPLIT]

    budgets = {
        "global_conformal": {q: conf["q_hat_global"] for q in qids},
        "bins3": {q: conf["q_hat_bins"][int(assign_bins([t.loc[q, "conf_score"]], conf["edges"])[0])]
                  for q in qids},
        "cqr": dict(zip(qids, np.exp(qr.predict(t.loc[qids, SIGNALS]) + cqr["Q_log"]))),
    }
    print("pre-declared main method: global_conformal | q_hat =", conf["q_hat_global"])

    recs = []
    t0 = time.time()
    for method, bq in budgets.items():
        for q in qids:
            ids = ret[q]["ids"]
            fed = prefix_within(ids, tc, bq[q])
            resp, info = chat_checked(build_messages(rows[q]["question"], [chunks[c]["text"] for c in fed]),
                                      MAIN_OPTIONS)
            ans = resp["message"]["content"].strip()
            g = set(gold[q])
            recs.append({
                "query_id": q, "split": SPLIT, "method": method, "budget": round(float(bq[q]), 1),
                "k": len(fed), "fed_tokens": int(tc[fed].sum()), **info,
                "gold_in_pool": g <= set(ids), "gold_covered_both": g <= set(fed),
                "gold_covered_any": bool(g & set(fed)), "answer": ans, "gold_answer": rows[q]["answer"],
                "em": exact_match(ans, rows[q]["answer"]), "f1": round(f1_score(ans, rows[q]["answer"]), 4),
            })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} LLM calls in {time.time() - t0:.0f} s | saved {OUT}")

    # fixed-k retrieval curve on the same test queries (k = 1..20)
    curve = []
    for k in range(1, 21):
        both = [set(gold[q]) <= set(ret[q]["ids"][:k]) for q in qids]
        pool = [set(gold[q]) <= set(ret[q]["ids"]) for q in qids]
        curve.append({"k": k, "recall_both": np.mean(both),
                      "cov_conditional": np.mean([b for b, p in zip(both, pool) if p]),
                      "tokens_mean": np.mean([tc[ret[q]["ids"][:k]].sum() for q in qids])})
    pd.DataFrame(curve).round(4).to_csv("results/fixed_k_curve_test.csv", index=False)

    d = pd.DataFrame(recs)
    k20 = curve[-1]["tokens_mean"]
    print(f"\ntest: {len(qids)} queries, both gold in top 20: {int(d[d.method == 'global_conformal'].gold_in_pool.sum())}")
    print(f"{'method':<17} | {'recall both':>11} | {'cond cov':>8} | {'tokens':>7} | {'saved vs k20':>12} | "
          f"{'avg k':>5} | {'EM':>5} | {'F1':>5} | trunc")
    for m in budgets:
        x = d[d.method == m]
        print(f"{m:<17} | {x.gold_covered_both.mean():>11.3f} | {x[x.gold_in_pool].gold_covered_both.mean():>8.3f} | "
              f"{x.fed_tokens.mean():>7.1f} | {100 * (1 - x.fed_tokens.mean() / k20):>11.1f}% | "
              f"{x.k.mean():>5.2f} | {x.em.mean():>5.3f} | {x.f1.mean():>5.3f} | {int(x.truncated.sum())}")
    print("\nfixed-k curve on test (retrieval):")
    for c in curve:
        if c["k"] in (5, 8, 10, 11, 12, 13, 15, 20):
            print(f"  k={c['k']:>2}: recall both {c['recall_both']:.3f} | cond cov {c['cov_conditional']:.3f} | "
                  f"tokens {c['tokens_mean']:.1f}")
