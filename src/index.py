"""Phase 1 (1.4, 1.5): embed chunks, build a FAISS HNSW index, and search it.

Embeddings are L2-normalized. Distance = true L2 distance (sqrt of FAISS's squared L2).
For unit vectors: L2^2 = 2 - 2*cos, so a smaller distance = more similar.
"""
import json
import os

import faiss
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # starting value
HNSW_M = 32                # starting value
EF_CONSTRUCTION = 200      # starting value
EF_SEARCH = 100            # starting value

CHUNK_PATH = "data/processed/chunks.jsonl"
EMB_PATH = "data/processed/chunk_embeddings.npy"
INDEX_PATH = "data/processed/hnsw.index"


def load_chunks(path=CHUNK_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(model, texts, batch_size=64):
    """Return float32 array, L2-normalized."""
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype("float32")


def build_index(emb, M=HNSW_M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH):
    index = faiss.IndexHNSWFlat(emb.shape[1], M)  # default metric = L2
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(emb)
    return index


def search(index, model, query, top_m):
    """Return (chunk_ids, distances) for one query text. Distances are L2, ascending."""
    q = embed_texts(model, [query])
    return search_vec(index, q, top_m)


def search_vec(index, q_vecs, top_m):
    D2, I = index.search(q_vecs, top_m)
    return I, np.sqrt(np.maximum(D2, 0.0))


if __name__ == "__main__":
    chunks = load_chunks()
    model = get_model()
    print("embedding", len(chunks), "chunks ...")
    emb = embed_texts(model, [c["text"] for c in chunks])
    np.save(EMB_PATH, emb)
    norms = np.linalg.norm(emb, axis=1)
    print("embeddings:", emb.shape, "| norm min/max:", norms.min().round(4), norms.max().round(4))

    index = build_index(emb)
    faiss.write_index(index, INDEX_PATH)
    print("index built: ntotal =", index.ntotal, "| M =", HNSW_M,
          "| efConstruction =", EF_CONSTRUCTION, "| efSearch =", EF_SEARCH)
