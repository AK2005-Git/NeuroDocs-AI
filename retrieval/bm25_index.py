import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


INDEX_DIR = Path("data/indexes")

METADATA_PATH = INDEX_DIR / "metadata.pkl"

BM25_PATH = INDEX_DIR / "bm25.pkl"


print("Loading metadata...")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded {len(metadata)} chunks")


# Tokenize text

tokenized_corpus = []

for chunk in metadata:

    text = chunk["chunk_text"]

    tokens = text.lower().split()

    tokenized_corpus.append(tokens)


print("Building BM25 index...")

bm25 = BM25Okapi(tokenized_corpus)

with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index saved successfully!")