from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )
    document_id: Optional[str] = None


class SourceResponse(BaseModel):
    filename: str
    page_number: int
    similarity: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]

