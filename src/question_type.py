"""Option 3: question-type signal q_comparison = predicted P(question is a "comparison" question).

HotpotQA labels each question "comparison" (e.g. "Which is older, X or Y?") or "bridge".
Comparison questions name both entities, so both gold passages should be easy to retrieve.
The label is NOT available at query time, so we train a small text classifier on FIT questions
(TF-IDF of word 1-2 grams + logistic regression) and use its predicted probability.

- Fit rows: OUT-OF-FOLD prediction (5-fold), so no question is scored by a model that saw it.
- Calibration and test rows: classifier trained on all fit questions.
The true type is read from the cached HotpotQA download (data/raw is not changed).
Output: data/processed/question_type.csv (query_id, split, q_type_true, q_comparison)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from data_loader import load_raw

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
OUT = "data/processed/question_type.csv"


def make_clf():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True),
                         LogisticRegression(C=1.0, max_iter=1000))


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    type_of = dict(zip(ds["id"], ds["type"]))

    rows = {r["id"]: r for r in load_raw()}
    splits = json.load(open("data/processed/splits.json"))
    df = pd.DataFrame([{"query_id": q, "split": s, "question": rows[q]["question"],
                        "q_type_true": type_of[q]} for s, qs in splits.items() for q in qs])
    y = (df.q_type_true == "comparison").astype(int).to_numpy()
    fit = (df.split == "fit").to_numpy()

    p = np.zeros(len(df))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p[fit] = cross_val_predict(make_clf(), df.question[fit], y[fit], cv=cv, method="predict_proba")[:, 1]
    p[~fit] = make_clf().fit(df.question[fit], y[fit]).predict_proba(df.question[~fit])[:, 1]
    df["q_comparison"] = p
    df[["query_id", "split", "q_type_true", "q_comparison"]].to_csv(OUT, index=False)

    print("comparison share:", {s: round(float(y[(df.split == s).to_numpy()].mean()), 3)
                                for s in ("fit", "calibration")})
    print(f"classifier on fit (out-of-fold): AUROC {roc_auc_score(y[fit], p[fit]):.3f} | "
          f"accuracy {accuracy_score(y[fit], p[fit] > 0.5):.3f}")
    # does question type relate to tokens needed? (fit only)
    lab = pd.read_csv("data/processed/labels.csv")
    m = df[fit].merge(lab, on=["query_id", "split"])
    print("\nfit, by TRUE type:")
    print(m.groupby("q_type_true").agg(n=("query_id", "size"), coverable=("gold_in_pool", "mean"),
                                       median_tokens=("gold_pos_tokens", "median"),
                                       p90_tokens=("gold_pos_tokens", lambda x: x.quantile(0.9))).round(3).to_string())
    c = m[m.gold_in_pool]
    print(f"Spearman(q_comparison, gold_pos_tokens) on coverable fit: "
          f"{c.q_comparison.corr(c.gold_pos_tokens, method='spearman'):+.3f}")
    print("saved", OUT)
