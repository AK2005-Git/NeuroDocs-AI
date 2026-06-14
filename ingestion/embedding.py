import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Paths
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# Load Embedding Model
print("Loading BGE-M3 model...")

model = SentenceTransformer("BAAI/bge-m3")

print("Model loaded successfully!\n")


def process_chunk_file(chunk_file):

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["chunk_text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    output_data = {
        "metadata": chunks,
        "embeddings": embeddings
    }

    output_file = EMBEDDINGS_DIR / f"{chunk_file.stem}.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved: {output_file.name}")


def process_all_chunk_files():

    chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))

    if not chunk_files:
        print("No chunk files found.")
        return

    print(f"Found {len(chunk_files)} chunk files.\n")

    for chunk_file in chunk_files:

        print("=" * 60)
        print(f"Processing: {chunk_file.name}")

        process_chunk_file(chunk_file)

    print("\n" + "=" * 60)
    print("Embedding generation completed!")
    print("=" * 60)
import pickle

with open(
    "data/embeddings/FNH unit 1,2,3_chunks.pkl",
    "rb"
) as f:
    data = pickle.load(f)

print(data["embeddings"].shape)

if __name__ == "__main__":
    process_all_chunk_files()