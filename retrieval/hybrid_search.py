import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# PATHS
# ==========================================

INDEX_DIR = Path("data/indexes")

FAISS_PATH = INDEX_DIR / "rag_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
BM25_PATH = INDEX_DIR / "bm25.pkl"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading BGE-M3 model...")
model = SentenceTransformer("BAAI/bge-m3")

print("Loading FAISS index...")
faiss_index = faiss.read_index(str(FAISS_PATH))

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading BM25 index...")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("\nSystem Ready!")
print(f"Total Chunks: {len(metadata)}")


# ==========================================
# FAISS SEARCH
# ==========================================

def faiss_search(query, top_k=5):

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

    results = []

    for rank, idx in enumerate(indices[0], start=1):

        if idx == -1:
            continue

        chunk = metadata[idx]

        results.append({
            "rank": rank,
            "source_file": chunk["source_file"],
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "method": "FAISS"
        })

    return results


# ==========================================
# BM25 SEARCH
# ==========================================

def bm25_search(query, top_k=5):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        chunk = metadata[idx]

        results.append({
            "rank": rank,
            "source_file": chunk["source_file"],
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "method": "BM25"
        })

    return results


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(title, results):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    for result in results:

        print(f"\nRank        : {result['rank']}")
        print(f"Method      : {result['method']}")
        print(f"Source File : {result['source_file']}")
        print(f"Chunk ID    : {result['chunk_id']}")

        print("\nPreview:")
        print("-" * 100)

        print(result["chunk_text"][:500])

        print("\n" + "=" * 100)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk a question (exit to quit): ").strip()

        if query.lower() == "exit":
            break

        faiss_results = faiss_search(query, top_k=5)

        bm25_results = bm25_search(query, top_k=5)

        display_results(
            "FAISS RESULTS",
            faiss_results
        )

        display_results(
            "BM25 RESULTS",
            bm25_results
        )