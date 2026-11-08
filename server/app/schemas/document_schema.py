from pydantic import BaseModel
from typing import List


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    total_pages: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str


class FolderIngestRequest(BaseModel):
    folder_path: str = "storage/pdf"
    recursive: bool = False


class FolderIngestResponse(BaseModel):
    message: str
    folder: str
    total_files_found: int
    successful_ingestions: int
    total_chunks: int
    processed_documents: List[dict]
    errors: List[dict]


