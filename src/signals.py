"""Phase 3 (3.1): confidence signals per query, computed from the retrieved distances only.

Input: the top_m distances of one query, ascending (smaller = closer), L2 on unit vectors.
- margin_mc = d2 - d1                      (gap between top-1 and top-2; subtraction, not ratio)
- var_local = (1/m) * sum((d_i - mean_d)^2) (population variance of the m distances)
- top1_dist = d1                           (rival signal to beat)
- elbow_idx = number of chunks before the biggest jump between neighbouring distances
- entropy   = entropy of softmax(-d / TAU), divided by log(m) so it is in [0, 1]
"""
import numpy as np

TOP_M = 20   # candidates per query (starting value)
TAU = 0.1    # softmax temperature for entropy (starting value; check in 3.4)


def compute_signals(dists, tau=TAU):
    d = np.sort(np.asarray(dists, dtype=float))
    m = len(d)
    if m < 2:
        raise ValueError("need at least 2 distances")

    margin_mc = d[1] - d[0]
    var_local = float(np.mean((d - d.mean()) ** 2))  # 1/m, not 1/(m-1)
    top1_dist = d[0]
    elbow_idx = int(np.argmax(np.diff(d))) + 1        # e.g. jump after rank 2 -> 2

    z = -d / tau
    p = np.exp(z - z.max())
    p /= p.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(m))

    return {
        "margin_mc": float(margin_mc),
        "var_local": var_local,
        "top1_dist": float(top1_dist),
        "elbow_idx": elbow_idx,
        "entropy": entropy,
    }


if __name__ == "__main__":
    # small demo on 3 FIT queries (not the test split)
    import json

    import faiss

    from data_loader import load_raw
    from index import INDEX_PATH, get_model, embed_texts, search_vec

    index = faiss.read_index(INDEX_PATH)
    model = get_model()
    splits = json.load(open("data/processed/splits.json"))
    rows = {r["id"]: r for r in load_raw()}
    qids = splits["fit"][:3]
    _, D = search_vec(index, embed_texts(model, [rows[q]["question"] for q in qids]), TOP_M)
    for q, d in zip(qids, D):
        s = compute_signals(d)
        print(rows[q]["question"][:70])
        print("   top 5 dists:", np.round(d[:5], 4))
        print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
