import pickle
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/indexes")

INDEX_DIR.mkdir(parents=True, exist_ok=True)


all_embeddings = []
all_metadata = []


embedding_files = list(EMBEDDINGS_DIR.glob("*.pkl"))

print(f"Found {len(embedding_files)} embedding files\n")


for file in embedding_files:

    print(f"Loading: {file.name}")

    with open(file, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    metadata = data["metadata"]

    all_embeddings.append(embeddings)
    all_metadata.extend(metadata)


all_embeddings = np.vstack(all_embeddings)

print("\nTotal Chunks:", len(all_metadata))
print("Embedding Shape:", all_embeddings.shape)


dimension = all_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(all_embeddings.astype("float32"))

print(f"\nFAISS vectors stored: {index.ntotal}")


faiss.write_index(
    index,
    str(INDEX_DIR / "rag_index.faiss")
)

with open(INDEX_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print("\nFAISS index saved successfully!")