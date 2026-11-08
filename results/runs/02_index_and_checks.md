# Run 02: Embeddings, HNSW index and index checks (tasks 1.4-1.7)

## Goal
Embed the chunks, build a FAISS HNSW index, check that search results make sense, and check how much HNSW loses compared with exact search.

## Setup
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dims), L2-normalized.
- Index: `faiss.IndexHNSWFlat`, M 32, efConstruction 200, efSearch 100 (starting values).
- Distance: L2 on unit vectors (square root of FAISS's squared L2), range [0, 2], smaller = closer.
- Checks on the **fit split only** (350 queries), so the test split is not looked at.
- Commands: `python src\index.py`, then `python notebooks\check_index.py`.

## What happened
- 4,956 embeddings, shape (4956, 384), every norm = 1.0.
- Hand check of 3 fit queries: both gold chunks came back at ranks 1 and 2 every time.
  - "Tamiami Canal or Genesee River?": Tamiami Canal (d 0.908), Genesee River (d 0.953).
  - "Palm Springs ... Palm Beach County": Palm Springs, Florida (d 0.490), Palm Beach County (d 0.778).
  - "Saab 36 and Avro Vulcan": Saab 36 (d 0.938), Avro Vulcan (d 0.945).
- Top-1 distance over fit queries: min 0.373, median 0.841, max 1.117.
- Gold found on fit:

| | top 5 | top 10 | top 20 |
|---|---|---|---|
| both gold | 0.520 | 0.683 | 0.763 |
| at least one gold | 0.971 | 0.991 | 0.994 |

- HNSW vs exact search, top 20: mean recall **0.9994** (min 0.95). Top-1 identical for 100% of queries.

## Good findings
- **HNSW loses almost nothing.** 0.9994 recall means Risk 5 ("HNSW is approximate") is not a problem at this size. No need to tune HNSW.
- Results look sensible by eye: the right articles come first.

## Surprising
- **The first gold is easy; the second is hard.** At top 5, 97% of queries have one gold but only 52% have both. Multi-hop questions often need a second passage that does not look like the question (the "bridge" passage).
- Distances are high overall (median top-1 0.84 out of a max of 2). MiniLM is a small model, so even good matches are not very close.

## Stagnant / weak
- Both-gold recall grows slowly: +0.163 from top 5 to top 10, then only +0.080 from top 10 to top 20. Each extra chunk adds less.
- 24% of fit queries do not have both gold even in the top 20. Budgeting cannot fix these, because the gold is not in the pool at all.

## What went wrong
- Nothing broke.
- One limitation was found: **188 of 4,956 chunks are longer than MiniLM's 256-token limit**, so their end is cut off before embedding. If the answer sits at the end of a long paragraph, that chunk may rank too low.

## Next
- The "both vs any gold" gap is the main question for task 4.1 (how to count multi-hop gold).
