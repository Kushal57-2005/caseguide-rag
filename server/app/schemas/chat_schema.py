from pydantic import BaseModel, Field
from typing import List, Optional


class HistoryMessage(BaseModel):
    role: str
    content: str


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
    conversation_history: Optional[List[HistoryMessage]] = Field(default_factory=list)


class QueryInfo(BaseModel):
    original_query: str
    refined_query: str
    plain_language_query: str


class ProblemSummary(BaseModel):
    title: str
    category: str
    description: str


class AnswerDetails(BaseModel):
    summary: str
    explanation: List[str] = Field(default_factory=list)


class CitationItem(BaseModel):
    citation_id: str
    act: str
    section: str
    title: str
    supporting_text: str
    source_id: str
    file_name: str
    page_number: int


class RetrievalChunkItem(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    raw_distance: float
    similarity_score: Optional[float] = None
    reranker_score: Optional[float] = None
    retrieval_rank: int
    content: str


class RetrievalDetails(BaseModel):
    total_chunks_retrieved: int
    chunks: List[RetrievalChunkItem] = Field(default_factory=list)


class ActionStep(BaseModel):
    step: int
    title: str
    description: str
    source_ids: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    source_type: str = "RAG_AND_LLM"


class NextSteps(BaseModel):
    actions: List[ActionStep] = Field(default_factory=list)


class WarningItem(BaseModel):
    type: str
    message: str


class ResponseMetadata(BaseModel):
    status: str
    model: str
    retrieval_method: str
    top_k: int


class StructuredRAGResponse(BaseModel):
    query: QueryInfo
    problem_summary: ProblemSummary
    answer: AnswerDetails
    citations: List[CitationItem] = Field(default_factory=list)
    retrieval: RetrievalDetails
    next_steps: NextSteps
    warnings: List[WarningItem] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    metadata: ResponseMetadata


# Keep alias for compatibility
ChatResponse = StructuredRAGResponse


