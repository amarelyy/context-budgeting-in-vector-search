"""Phase 4 (4.1-4.3): labels per query.

Multi-hop rule (4.1): "covered" = BOTH gold chunks are inside the fed set.
gold_pos_tokens = sum of token_cost of the ranked chunks up to and including the
                  LAST gold chunk (= the smallest token budget that covers both gold).
gold_pos_rank   = rank (1-based) of that last gold chunk.

Not found rule (4.2): if a gold chunk is not in the top_m = 20 pool, the query can never
be covered by any budget. Then gold_in_pool = False and gold_pos_tokens / gold_pos_rank
are empty (NaN = "infinite"). These queries are KEPT (dropping them would hide failures).

Extra columns for analysis (not the main label):
first_gold_tokens = tokens up to the FIRST gold chunk (the "any gold" view)
pool_tokens       = tokens of all 20 chunks (= fixed k = 20 cost)
"""
import json

import numpy as np
import pandas as pd

TOP_M = 20
LABELS_PATH = "data/processed/labels.csv"


def label_query(ids, gold_ids, token_cost):
    ids = list(ids)[:TOP_M]
    cum = np.cumsum(token_cost[ids])
    pos = sorted(ids.index(g) for g in gold_ids if g in ids)
    in_pool = len(pos) == len(gold_ids)
    return {
        "gold_in_pool": in_pool,
        "n_gold_in_pool": len(pos),
        "gold_pos_rank": pos[-1] + 1 if in_pool else np.nan,
        "gold_pos_tokens": int(cum[pos[-1]]) if in_pool else np.nan,
        "first_gold_tokens": int(cum[pos[0]]) if pos else np.nan,
        "pool_tokens": int(cum[-1]),
    }


if __name__ == "__main__":
    chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl", encoding="utf-8")]
    token_cost = np.array([c["token_cost"] for c in chunks])
    ret = json.load(open("data/processed/retrieval_top20.json"))
    gold = json.load(open("data/processed/gold_chunk_ids.json"))
    splits = json.load(open("data/processed/splits.json"))

    rows = []
    for split, qids in splits.items():
        for q in qids:
            rows.append({"query_id": q, "split": split,
                         **label_query(ret[q]["ids"], gold[q], token_cost)})
    df = pd.DataFrame(rows)
    df.to_csv(LABELS_PATH, index=False)
    print("saved", len(df), "rows to", LABELS_PATH)

    # summary: fit + calibration only (test labels are saved for evaluation, not looked at here)
    for split in ("fit", "calibration"):
        d = df[df.split == split]
        c = d[d.gold_in_pool]
        print(f"\n[{split}] n = {len(d)} | both gold in pool: {len(c)} ({len(c) / len(d):.3f}) | "
              f"1 gold: {(d.n_gold_in_pool == 1).sum()} | 0 gold: {(d.n_gold_in_pool == 0).sum()}")
        print(f"  gold_pos_tokens (coverable): mean {c.gold_pos_tokens.mean():.1f} | "
              f"median {c.gold_pos_tokens.median():.1f} | p90 {c.gold_pos_tokens.quantile(0.9):.1f} | "
              f"max {c.gold_pos_tokens.max():.0f}")
        print(f"  gold_pos_rank   (coverable): median {c.gold_pos_rank.median():.0f} | "
              f"rank<=2: {(c.gold_pos_rank <= 2).mean():.3f} | rank<=5: {(c.gold_pos_rank <= 5).mean():.3f}")
        print(f"  pool_tokens (k=20) mean {d.pool_tokens.mean():.1f}")
    print("\n[test] rows saved:", (df.split == "test").sum(), "(summary not shown)")
