from ingestion.pdf_loader import (
    process_all_pdfs
)

from ingestion.chunker import (
    process_all_text_files
)

from ingestion.embedder import (
    build_index
)


def rebuild_rag():

    print("\n" + "=" * 60)
    print("REBUILDING RAG PIPELINE")
    print("=" * 60)

    process_all_pdfs()

    process_all_text_files()

    build_index()

    print("\nRAG Rebuild Complete")