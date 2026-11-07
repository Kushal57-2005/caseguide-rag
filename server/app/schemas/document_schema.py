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

