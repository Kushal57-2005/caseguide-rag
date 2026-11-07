import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.document_schema import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentDeleteResponse,
)
from app.services.ingestion_service import ingest_pdf_file
from app.services.vector_store import list_documents, delete_document

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)

STORAGE_DIR = Path("storage/pdf")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    saved_path = STORAGE_DIR / file.filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    try:
        ingest_result = ingest_pdf_file(
            file_path=str(saved_path),
            original_filename=file.filename
        )
        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully.",
            document_id=ingest_result["document_id"],
            filename=ingest_result["filename"],
            total_pages=ingest_result["total_pages"],
            total_chunks=ingest_result["total_chunks"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index PDF: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
def get_documents():
    try:
        raw_docs = list_documents()
        docs = [
            DocumentInfo(
                document_id=d["document_id"],
                filename=d["filename"],
                total_chunks=d["total_chunks"],
                total_pages=d["total_pages"]
            )
            for d in raw_docs
        ]
        return DocumentListResponse(
            documents=docs,
            total_documents=len(docs)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def remove_document(document_id: str):
    try:
        delete_document(document_id)
        return DocumentDeleteResponse(
            message="Document removed from vector store.",
            document_id=document_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

