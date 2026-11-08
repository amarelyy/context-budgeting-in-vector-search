"""Phase 3 (3.4): histograms of the signals. FIT and CALIBRATION only (test is not looked at).

Also checks the multi-hop margin_mc observation from 3.1, on the fit split:
are queries with BOTH gold at ranks 1-2 given a small margin_mc?
Run from the repo root:  .\.venv\Scripts\python.exe notebooks\plot_signals.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SIG = ["margin_mc", "var_local", "top1_dist", "elbow_idx", "entropy"]
OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/signals.csv")
df = df[df["split"].isin(["fit", "calibration"])]

# ---- summary table ----
print("=== summary (fit + calibration) ===")
print(df.groupby("split")[SIG].describe().T.round(4).to_string())

# ---- correlation between signals (fit only): are some redundant? ----
print("\n=== Spearman correlation between signals (fit) ===")
print(df[df.split == "fit"][SIG].corr(method="spearman").round(2).to_string())

# ---- histograms ----
fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
for ax, s in zip(axes, SIG):
    for split, color in (("fit", "#3b6fb6"), ("calibration", "#d08a2e")):
        x = df[df.split == split][s]
        bins = range(1, 21) if s == "elbow_idx" else 30
        ax.hist(x, bins=bins, alpha=0.6, density=True, color=color, label=split)
    ax.set_title(s)
axes[0].legend()
fig.suptitle("Signal distributions (fit n=350, calibration n=75; test not shown)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/signals_hist.png", dpi=110)

# ---- multi-hop check on fit: margin_mc when both gold are at ranks 1-2 ----
gold = json.load(open("data/processed/gold_chunk_ids.json"))
ret = json.load(open("data/processed/retrieval_top20.json"))
fit = df[df.split == "fit"].copy()
fit["both_gold_top2"] = [set(gold[q]) == set(ret[q]["ids"][:2]) for q in fit.query_id]
print("\n=== margin_mc on fit, by 'both gold at ranks 1-2' ===")
print(fit.groupby("both_gold_top2")["margin_mc"].describe().round(4).to_string())

fig, ax = plt.subplots(figsize=(6, 3.6))
for flag, color, lab in ((True, "#2e8b57", "both gold at ranks 1-2"),
                         (False, "#999999", "other queries")):
    ax.hist(fit[fit.both_gold_top2 == flag]["margin_mc"], bins=30, alpha=0.6,
            density=True, color=color, label=f"{lab} (n={int((fit.both_gold_top2 == flag).sum())})")
ax.set_xlabel("margin_mc"); ax.legend(); ax.set_title("margin_mc on fit split")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/margin_multihop_fit.png", dpi=110)
print("\nplots saved to", OUT_DIR)
