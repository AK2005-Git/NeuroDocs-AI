import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Folder Paths
EXTRACTED_DIR = Path("data/extracted")
CHUNKS_DIR = Path("data/chunks")

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# Chunk Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_document(txt_file):

    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks = splitter.split_text(text)

    chunk_data = []

    for idx, chunk in enumerate(chunks):

        chunk_record = {
            "chunk_id": idx,
            "source_file": txt_file.stem,
            "chunk_text": chunk,
            "chunk_length": len(chunk)
        }

        chunk_data.append(chunk_record)

    return chunk_data


def process_all_text_files():

    txt_files = list(EXTRACTED_DIR.glob("*.txt"))

    if not txt_files:
        print("No TXT files found.")
        return

    total_chunks = 0

    for txt_file in txt_files:

        print("=" * 60)
        print(f"Processing: {txt_file.name}")

        chunk_data = chunk_document(txt_file)

        chunk_count = len(chunk_data)
        total_chunks += chunk_count

        output_file = CHUNKS_DIR / f"{txt_file.stem}_chunks.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=4, ensure_ascii=False)

        print(f"Chunks Created : {chunk_count}")
        print(f"Saved          : {output_file.name}")

    print("\n" + "=" * 60)
    print(f"Total Chunks Created: {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    process_all_text_files()