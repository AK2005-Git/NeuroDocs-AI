import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# PATHS
# =========================

INDEX_DIR = Path("data/indexes")

FAISS_INDEX_PATH = INDEX_DIR / "rag_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"


# =========================
# LOAD MODEL
# =========================

print("=" * 60)
print("Loading BGE-M3 model...")
print("=" * 60)

model = SentenceTransformer("BAAI/bge-m3")

print("Model loaded successfully!\n")


# =========================
# LOAD FAISS INDEX
# =========================

print("=" * 60)
print("Loading FAISS Index...")
print("=" * 60)

index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"Vectors Loaded : {index.ntotal}")
print(f"Metadata Loaded: {len(metadata)}")
print("\nSystem Ready!")
print("=" * 60)


# =========================
# SEARCH FUNCTION
# =========================

def semantic_search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (idx, distance) in enumerate(
        zip(indices[0], distances[0]),
        start=1
    ):

        if idx == -1:
            continue

        chunk = metadata[idx]

        results.append(
            {
                "rank": rank,
                "score": float(distance),
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "chunk_length": chunk["chunk_length"],
                "chunk_text": chunk["chunk_text"]
            }
        )

    return results


# =========================
# DISPLAY FUNCTION
# =========================

def display_results(results):

    print("\n")
    print("=" * 100)
    print("TOP RETRIEVAL RESULTS")
    print("=" * 100)

    for result in results:

        print(f"\nRank         : {result['rank']}")
        print(f"Score        : {result['score']:.4f}")
        print(f"Source File  : {result['source_file']}")
        print(f"Chunk ID     : {result['chunk_id']}")
        print(f"Chunk Length : {result['chunk_length']}")

        print("\nChunk Content:")
        print("-" * 100)

        preview = result["chunk_text"][:1000]

        print(preview)

        print("\n" + "=" * 100)


# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":

    print("\nSemantic Search Ready")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        try:

            results = semantic_search(
                query=query,
                top_k=5
            )

            display_results(results)

        except Exception as e:

            print("\nError occurred:")
            print(e)