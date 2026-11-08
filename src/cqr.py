"""Phase 6, option 2: conformalized quantile regression (CQR), one-sided, on a log scale.

Instead of bins, predict each query's needed tokens directly, then add ONE conformal
correction computed from all coverable calibration queries (no per-bin penalty).

1. On FIT (coverable queries): linear quantile regression of log(gold_pos_tokens) on the
   signals, at quantile 1 - alpha.
2. On CALIBRATION (coverable): score_i = log(gold_pos_tokens_i) - prediction_i.
   Q = conformal quantile of the scores (finite-sample correction).
3. Budget for a query = exp(prediction + Q).
Method A (conditional coverage), alpha 0.1, as decided.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal import ALPHA, conformal_qhat

SIGNALS = ["margin_mc", "var_local", "top1_dist", "elbow_idx", "entropy"]
QR_PENALTY = 0.0     # no L1 penalty (starting value): only 5-6 features, 267 fit rows


def make_qr(alpha=ALPHA):
    return make_pipeline(StandardScaler(),
                         QuantileRegressor(quantile=1 - alpha, alpha=QR_PENALTY, solver="highs"))


def fit_qr(X, gold_pos_tokens):
    return make_qr().fit(X, np.log(gold_pos_tokens))


def calibrate(pred_log_cal, gold_cal, alpha=ALPHA):
    return conformal_qhat(np.log(gold_cal) - pred_log_cal, alpha)


def budgets(pred_log, Q):
    return np.exp(pred_log + Q)


def load_table(extra_features_path=None):
    s = pd.read_csv("data/processed/signals.csv")
    lab = pd.read_csv("data/processed/labels.csv")
    df = s.merge(lab, on=["query_id", "split"])
    if extra_features_path:
        df = df.merge(pd.read_csv(extra_features_path), on=["query_id", "split"])
    return df


if __name__ == "__main__":
    import sys
    feats = SIGNALS + (["q_comparison"] if "--qtype" in sys.argv else [])
    df = load_table("data/processed/question_type.csv" if "--qtype" in sys.argv else None)
    tag = "cqr_qtype" if "--qtype" in sys.argv else "cqr"

    fit = df[(df.split == "fit") & df.gold_in_pool]
    cal = df[df.split == "calibration"]
    calc = cal[cal.gold_in_pool]

    qr = fit_qr(fit[feats], fit.gold_pos_tokens)
    Q = calibrate(qr.predict(calc[feats]), calc.gold_pos_tokens)
    b = budgets(qr.predict(cal[feats]), Q)
    w = qr.named_steps["quantileregressor"].coef_
    print(f"features: {feats}")
    print("QR coefficients (standardized, log scale): " + ", ".join(f"{f} {c:+.3f}" for f, c in zip(feats, w)))
    print(f"fit coverable n = {len(fit)} | calibration coverable n = {len(calc)} | Q = {Q:+.3f} (log scale)")
    print(f"budgets on calibration: min {b.min():.0f} | median {np.median(b):.0f} | max {b.max():.0f} | mean {b.mean():.0f}")
    pool = cal.gold_in_pool.to_numpy()
    cov_c = (calc.gold_pos_tokens.to_numpy() <= b[pool]).mean()
    print(f"in-sample coverage on calibration (conditional): {cov_c:.3f} | target {1 - ALPHA:.2f}")
    joblib.dump(qr, f"data/processed/{tag}_model.joblib")
    with open(f"data/processed/{tag}.json", "w") as f:
        json.dump({"alpha": ALPHA, "features": feats, "Q_log": Q, "qr_penalty": QR_PENALTY}, f, indent=1)
    print(f"saved data/processed/{tag}.json and model")
