"""Phase 1 (1.3): make chunks from the passages and compute token_cost.

One paragraph = one chunk. All queries share one corpus (same paragraph = one chunk).
token_cost = number of tokens, counted with the tokenizer below (starting value).
"""
import json
import os

from data_loader import load_raw

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # starting value
CHUNK_PATH = "data/processed/chunks.jsonl"
GOLD_PATH = "data/processed/gold_chunk_ids.json"


def build_chunks(rows, tokenizer):
    chunks, key_to_id = [], {}
    gold = {}  # query id -> list of gold chunk ids
    for r in rows:
        gold[r["id"]] = []
        for p in r["paragraphs"]:
            text = " ".join(s.strip() for s in p["sentences"])
            key = (p["title"], text)
            if key not in key_to_id:
                key_to_id[key] = len(chunks)
                n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                chunks.append({"chunk_id": key_to_id[key], "title": p["title"],
                               "text": text, "token_cost": n_tok})
            if p["title"] in r["gold_titles"]:
                gold[r["id"]].append(key_to_id[key])
    return chunks, gold


def save(chunks, gold):
    os.makedirs(os.path.dirname(CHUNK_PATH), exist_ok=True)
    with open(CHUNK_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(gold, f)


if __name__ == "__main__":
    import statistics
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    rows = load_raw()
    chunks, gold = build_chunks(rows, tok)
    save(chunks, gold)

    costs = [c["token_cost"] for c in chunks]
    print("queries:", len(rows), "| chunks:", len(chunks))
    print("token_cost  min/median/mean/max:",
          min(costs), statistics.median(costs), round(statistics.mean(costs), 1), max(costs))
    print("total tokens in corpus:", sum(costs))
    n_gold = [len(v) for v in gold.values()]
    print("gold chunks per query  min/max:", min(n_gold), max(n_gold),
          "| queries with 0 gold:", sum(1 for n in n_gold if n == 0))
    c = chunks[0]
    print("\nexample chunk:", c["chunk_id"], c["title"], "| token_cost =", c["token_cost"])
    print(c["text"][:150], "...")
