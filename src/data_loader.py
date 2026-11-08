"""Phase 1 (1.1, 1.2): load a small HotpotQA subset and make fit/calibration/test splits."""
import json
import os
import random

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SEED = 42            # fixed seed (starting value)
N_QUERIES = 500      # subset size (starting value)
SPLIT_RATIO = (0.70, 0.15, 0.15)  # fit / calibration / test (starting value)

RAW_PATH = "data/raw/hotpot_subset.jsonl"
SPLIT_PATH = "data/processed/splits.json"


def load_subset(n=N_QUERIES, seed=SEED):
    """Download HotpotQA (distractor, validation) and keep n random queries."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), n))
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            # gold = titles of the supporting paragraphs
            "gold_titles": sorted(set(ex["supporting_facts"]["title"])),
            # 10 paragraphs per query: title + list of sentences
            "paragraphs": [
                {"title": t, "sentences": s}
                for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
            ],
        })
    return rows


def save_raw(rows, path=RAW_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_raw(path=RAW_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_splits(rows, ratio=SPLIT_RATIO, seed=SEED, path=SPLIT_PATH):
    """Split query IDs into fit / calibration / test. Save the IDs."""
    ids = [r["id"] for r in rows]
    random.Random(seed).shuffle(ids)
    n_fit = int(len(ids) * ratio[0])
    n_cal = int(len(ids) * ratio[1])
    splits = {
        "fit": ids[:n_fit],
        "calibration": ids[n_fit:n_fit + n_cal],
        "test": ids[n_fit + n_cal:],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    return splits


if __name__ == "__main__":
    if os.path.exists(RAW_PATH):
        print("Raw file exists, not overwriting:", RAW_PATH)
        rows = load_raw()
    else:
        rows = load_subset()
        save_raw(rows)
        print("Saved", len(rows), "queries to", RAW_PATH)

    ex = rows[0]
    print("\n--- one example ---")
    print("question:", ex["question"])
    print("answer:", ex["answer"])
    print("gold titles:", ex["gold_titles"])
    for p in ex["paragraphs"][:3]:
        print(" -", p["title"], "|", " ".join(p["sentences"])[:100], "...")

    splits = make_splits(rows)
    print("\nsplit sizes:", {k: len(v) for k, v in splits.items()})
