import pickle
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIG
# ==========================================

TOP_K = 10
RRF_K = 60

INDEX_DIR = Path("data/indexes")

FAISS_PATH = INDEX_DIR / "rag_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
BM25_PATH = INDEX_DIR / "bm25.pkl"


# ==========================================
# LOAD RESOURCES
# ==========================================

print("Loading BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

print("Loading FAISS...")
faiss_index = faiss.read_index(str(FAISS_PATH))

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading BM25...")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("\nSystem Ready")
print(f"Chunks Loaded: {len(metadata)}")


# ==========================================
# FAISS SEARCH
# ==========================================

def faiss_search(query, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    return indices[0].tolist()


# ==========================================
# BM25 SEARCH
# ==========================================

def bm25_search(query, top_k=TOP_K):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices.tolist()


# ==========================================
# RRF FUSION
# ==========================================

def reciprocal_rank_fusion(result_lists):

    rrf_scores = defaultdict(float)

    for result_list in result_lists:

        for rank, doc_id in enumerate(result_list):

            rrf_scores[doc_id] += 1.0 / (
                RRF_K + rank + 1
            )

    ranked_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_docs


# ==========================================
# HYBRID SEARCH
# ==========================================

def hybrid_search(query):

    faiss_results = faiss_search(query)

    bm25_results = bm25_search(query)

    fused_results = reciprocal_rank_fusion(
        [
            faiss_results,
            bm25_results
        ]
    )

    return fused_results[:5]


# ==========================================
# DISPLAY
# ==========================================

def display_results(results):

    print("\n")
    print("=" * 120)
    print("HYBRID RRF RESULTS")
    print("=" * 120)

    for rank, (doc_id, score) in enumerate(
        results,
        start=1
    ):

        chunk = metadata[doc_id]

        print(f"\nRank       : {rank}")
        print(f"RRF Score  : {score:.6f}")
        print(f"Source     : {chunk['source_file']}")
        print(f"Chunk ID   : {chunk['chunk_id']}")

        print("\nPreview:")
        print("-" * 120)

        print(chunk["chunk_text"][:700])

        print("\n" + "=" * 120)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk a question (exit to quit): "
        ).strip()

        if query.lower() == "exit":
            break

        results = hybrid_search(query)

        display_results(results)