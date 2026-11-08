"""Phase 5 (5.1-5.3): confidence score from the signals.

Target (5.1): y = 1 if BOTH gold chunks are covered within T tokens
              (gold_pos_tokens <= T), else 0. Uncoverable queries -> y = 0.
              T = median gold_pos_tokens of coverable FIT queries (starting value).
Model: logistic regression on the 5 signals (standardized, default L2, C = 1.0).
Rival (5.2): the same model on top1_dist only.
Output (5.3): conf_score (main) and conf_score_top1 (rival) for all splits.

Everything is fitted and judged on FIT only (5-fold CV). Calibration and test are only scored.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIGNALS = ["margin_mc", "var_local", "top1_dist", "elbow_idx", "entropy"]
RIVAL = ["top1_dist"]
SEED = 42
OUT_SCORES = "data/processed/conf_scores.csv"
OUT_MODEL = "data/processed/score_model.joblib"
OUT_MODEL_RIVAL = "data/processed/score_model_top1.joblib"


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def cv_auroc(X, y, seed=SEED):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


if __name__ == "__main__":
    s = pd.read_csv("data/processed/signals.csv")
    lab = pd.read_csv("data/processed/labels.csv")
    df = s.merge(lab, on=["query_id", "split"])
    fit = df[df.split == "fit"].copy()

    T = float(fit.loc[fit.gold_in_pool, "gold_pos_tokens"].median())
    df["y"] = (df.gold_pos_tokens <= T).astype(int)   # NaN (uncoverable) -> False -> 0
    fit = df[df.split == "fit"]
    print(f"T = {T:.0f} tokens (median gold_pos_tokens, coverable fit) | "
          f"fit positives: {fit.y.sum()} / {len(fit)} ({fit.y.mean():.3f})")

    # ---- 5-fold CV AUROC on fit (main vs rival vs each single signal) ----
    print("\n5-fold CV AUROC on fit (5 seeds for the split, mean ± std):")
    for name, cols in [("all 5 signals", SIGNALS), ("top1_dist only (rival)", RIVAL)] + \
                      [(c, [c]) for c in SIGNALS if c != "top1_dist"]:
        a = [cv_auroc(fit[cols].to_numpy(), fit.y.to_numpy(), seed) for seed in range(5)]
        print(f"  {name:<24} {np.mean(a):.3f} ± {np.std(a):.3f}")

    # sensitivity to T (info only, still fit-only)
    print("\nSensitivity to T (all 5 signals, CV AUROC, seed 42):")
    for q in (0.25, 0.5, 0.75):
        t = float(fit.loc[fit.gold_in_pool, "gold_pos_tokens"].quantile(q))
        yt = (fit.gold_pos_tokens <= t).astype(int)
        print(f"  T = p{int(q * 100)} = {t:.0f} tokens | positives {yt.mean():.3f} | "
              f"AUROC {cv_auroc(fit[SIGNALS].to_numpy(), yt.to_numpy()):.3f}")
    yp = fit.gold_in_pool.astype(int)
    print(f"  target = gold_in_pool       | positives {yp.mean():.3f} | "
          f"AUROC {cv_auroc(fit[SIGNALS].to_numpy(), yp.to_numpy()):.3f}")

    # ---- fit final models on all fit queries ----
    main = make_model().fit(fit[SIGNALS], fit.y)
    rival = make_model().fit(fit[RIVAL], fit.y)
    coef = main.named_steps["logisticregression"].coef_[0]
    print("\nMain model coefficients (standardized signals):")
    for c, w in zip(SIGNALS, coef):
        print(f"  {c:<10} {w:+.3f}")
    joblib.dump(main, OUT_MODEL)
    joblib.dump(rival, OUT_MODEL_RIVAL)

    # ---- 5.3 conf_score for all splits ----
    df["conf_score"] = main.predict_proba(df[SIGNALS])[:, 1]
    df["conf_score_top1"] = rival.predict_proba(df[RIVAL])[:, 1]
    df[["query_id", "split", "y", "conf_score", "conf_score_top1"]].to_csv(OUT_SCORES, index=False)
    with open("data/processed/score_model_meta.json", "w") as f:
        json.dump({"T": T, "signals": SIGNALS, "rival": RIVAL, "C": 1.0, "seed": SEED}, f)
    print(f"\nsaved {OUT_SCORES} ({len(df)} rows), models, meta")
    print("conf_score on fit: min %.3f | median %.3f | max %.3f" % (
        df.loc[df.split == "fit", "conf_score"].min(), df.loc[df.split == "fit", "conf_score"].median(),
        df.loc[df.split == "fit", "conf_score"].max()))
