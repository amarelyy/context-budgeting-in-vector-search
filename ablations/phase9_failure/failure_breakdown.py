"""Phase 9.3: failure analysis on the TEST split (reporting only; nothing is chosen from it).

Every test query under the main method (global conformal budget) falls in exactly one stage:
  1 retrieval_miss : a gold chunk is not in the top 20 (no budget can help)
  2 budget_miss    : both gold in the top 20, but the budget stopped before the last gold
  3 llm_fail       : both gold fed, answer F1 = 0
  4 partial        : both gold fed, 0 < F1 < 1
  5 correct        : both gold fed, F1 = 1
Also: observational position check on the fixed k = 20 run (F1 by the rank of the deepest gold),
and answer F1 for queries where gold was NOT fully fed (what the LLM gets from 1 gold / memory).
Output: failure_breakdown_test.csv, llm_failures_test.csv
"""
import pandas as pd

b = pd.read_csv("results/budget_test_main_seed42.csv")
g = b[b.method == "global_conformal"].copy()


def stage(r):
    if not r.gold_in_pool:
        return "1_retrieval_miss"
    if not r.gold_covered_both:
        return "2_budget_miss"
    if r.f1 == 0:
        return "3_llm_fail"
    return "4_partial" if r.f1 < 1 else "5_correct"


g["stage"] = g.apply(stage, axis=1)
t = g.groupby("stage").agg(n=("query_id", "size"), mean_f1=("f1", "mean"), mean_tokens=("fed_tokens", "mean"),
                           mean_chunks=("k", "mean")).round(3)
t["share"] = (t.n / len(g)).round(3)
t.to_csv("ablations/phase9_failure/failure_breakdown_test.csv")
print("=== stages, global budget, test (75) ===")
print(t.to_string())
print(f"\nF1 when gold NOT fully fed (stages 1+2): {g[g.stage.isin(['1_retrieval_miss', '2_budget_miss'])].f1.mean():.3f}"
      f" | when fully fed (3-5): {g[g.gold_covered_both].f1.mean():.3f}")
print(f"any gold fed among stage 1: {g[g.stage == '1_retrieval_miss'].gold_covered_any.mean():.3f}")

fail = g[g.stage == "3_llm_fail"][["query_id", "answer", "gold_answer", "k", "fed_tokens"]]
fail.to_csv("ablations/phase9_failure/llm_failures_test.csv", index=False)
print(f"\n=== stage 3, LLM failures with both gold fed (n={len(fail)}) ===")
for _, r in fail.iterrows():
    print(f"  pred {r.answer[:55]!r:<58} | gold {r.gold_answer[:40]!r}")

# observational position check on fixed k = 20 (both gold fed)
base = pd.read_csv("results/llm_baseline_test_main_seed42.csv")
k20 = base[(base.k == 20) & base.gold_covered_both].copy()
k20["deepest_gold_rank"] = k20.gold_ranks.str.split().apply(lambda x: max(map(int, x)))
k20["depth"] = pd.cut(k20.deepest_gold_rank, [0, 2, 5, 10, 20], labels=["1-2", "3-5", "6-10", "11-20"])
print("\n=== fixed k=20, both gold fed: F1 by rank of the deepest gold (observational, test) ===")
print(k20.groupby("depth", observed=True).f1.agg(["count", "mean"]).round(3).to_string())
