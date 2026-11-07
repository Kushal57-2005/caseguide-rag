from fastapi import APIRouter, HTTPException
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse
)
from app.services.retrieval_service import retrieve_documents
from app.services.generation_service import generate_answer

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieve_documents(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id
        )

        # Step 2: Generate answer
        result = generate_answer(
            question=request.question,
            retrieved_chunks=retrieved_chunks
        )

        # Step 3: Return response
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(error)}"
        )

