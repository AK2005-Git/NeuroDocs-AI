from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

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