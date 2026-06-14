# ingestion/embedder.py

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =====================================================
# PATHS
# =====================================================

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/indexes")

INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "rag_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

# =====================================================
# MODEL
# =====================================================

print("Loading BGE-M3...")

model = SentenceTransformer(
    "BAAI/bge-m3"
)

# =====================================================
# BUILD INDEX
# =====================================================

def build_index():

    all_chunks = []

    json_files = list(
        CHUNKS_DIR.glob("*_chunks.json")
    )

    if not json_files:

        print("No chunks found.")
        return

    for file in json_files:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

            all_chunks.extend(chunks)

    print(
        f"Total Chunks Loaded: {len(all_chunks)}"
    )

    texts = [
        c["chunk_text"]
        for c in all_chunks
    ]

    print("Generating Embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    print("FAISS Saved")

    with open(
        METADATA_PATH,
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    print("Metadata Saved")

    tokenized_docs = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )

    with open(
        BM25_PATH,
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )

    print("BM25 Saved")
    print("Index Build Complete")


if __name__ == "__main__":
    build_index()