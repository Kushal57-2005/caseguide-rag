from typing import Union
from fastapi import APIRouter, HTTPException
from app.schemas.chat_schema import (
    ChatRequest,
    StructuredRAGResponse,
    FallbackResponse
)
from app.services.rag_service import execute_rag_pipeline

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


@router.post("/", response_model=Union[StructuredRAGResponse, FallbackResponse])
def chat(request: ChatRequest):
    try:
        return execute_rag_pipeline(request)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(error)}"
        )
