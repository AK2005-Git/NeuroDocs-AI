from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

import shutil
from pathlib import Path

from ingestion.rebuild_pipeline import (
    rebuild_rag
)

router = APIRouter()

# ============================================================
# PDF STORAGE
# ============================================================

RAW_PDF_DIR = Path(
    "data/raw_pdfs"
)

RAW_PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# UPLOAD API
# ============================================================

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    try:

        if not file.filename.endswith(
            ".pdf"
        ):

            return {
                "status": "error",
                "message":
                "Only PDF files are allowed"
            }

        save_path = (
            RAW_PDF_DIR /
            file.filename
        )

        with open(
            save_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"\nPDF Uploaded: "
            f"{file.filename}"
        )

        # ===================================
        # AUTO REBUILD RAG
        # ===================================

        rebuild_rag()

        return {
            "status": "success",
            "message":
            "PDF uploaded and indexed successfully",
            "filename":
            file.filename
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# ============================================================
# LIST DOCUMENTS API
# ============================================================
# Returns the real list of PDFs currently in the index, read
# directly from disk — this is the source of truth, not the
# browser's session memory.

@router.get("/documents")
async def list_documents():

    try:

        pdf_files = sorted(
            RAW_PDF_DIR.glob("*.pdf")
        )

        documents = []

        for pdf_path in pdf_files:

            stat = pdf_path.stat()

            documents.append(
                {
                    "filename": pdf_path.name,
                    "size_bytes": stat.st_size
                }
            )

        return {
            "status": "success",
            "documents": documents,
            "count": len(documents)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
            "documents": [],
            "count": 0
        }

# ============================================================
# DELETE DOCUMENT API
# ============================================================
# Deletes the raw PDF from disk, then rebuilds the entire RAG
# pipeline so FAISS + BM25 + metadata regenerate without it.
# This is safe because rebuild_rag() always re-derives the
# full index from whatever PDFs remain in RAW_PDF_DIR — there
# is no manual/partial FAISS vector removal involved.

@router.delete("/documents/{filename}")
async def delete_document(
    filename: str
):

    try:

        target_path = (
            RAW_PDF_DIR / filename
        )

        # Prevent path traversal (e.g. "../../etc/passwd")
        if target_path.resolve().parent != RAW_PDF_DIR.resolve():

            raise HTTPException(
                status_code=400,
                detail="Invalid filename"
            )

        if not target_path.exists():

            raise HTTPException(
                status_code=404,
                detail=f"'{filename}' not found"
            )

        target_path.unlink()

        print(
            f"\nPDF Deleted: {filename}"
        )

        # ===================================
        # AUTO REBUILD RAG (without this file)
        # ===================================

        rebuild_rag()

        return {
            "status": "success",
            "message":
            f"'{filename}' deleted and index rebuilt successfully"
        }

    except HTTPException:

        raise

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }