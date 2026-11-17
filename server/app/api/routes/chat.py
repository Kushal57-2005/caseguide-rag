from fastapi import APIRouter, HTTPException
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    StructuredRAGResponse
)
from app.services.query_understanding import understand_query
from app.services.retrieval_service import retrieve_documents
from app.services.generation_service import generate_structured_response

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


@router.post("/", response_model=StructuredRAGResponse)
def chat(request: ChatRequest):
    try:
        # Step 1: Query Understanding stage with conversation history
        query_understanding = understand_query(
            original_query=request.question,
            conversation_history=request.conversation_history
        )
        search_query = query_understanding.get("refined_query") or request.question

        # Step 2: Retrieve relevant chunks using refined query with 0.4 similarity threshold
        retrieved_chunks = retrieve_documents(
            question=search_query,
            top_k=request.top_k,
            min_similarity=0.4,
            document_id=request.document_id
        )

        # Fallback to original question if refined query returned nothing with 0.4 threshold
        if not retrieved_chunks and search_query != request.question:
            retrieved_chunks = retrieve_documents(
                question=request.question,
                top_k=request.top_k,
                min_similarity=0.4,
                document_id=request.document_id
            )

        # Step 3: Generate structured response combining LLM + RAG + conversation history
        structured_response = generate_structured_response(
            original_query=request.question,
            query_understanding=query_understanding,
            retrieved_chunks=retrieved_chunks,
            top_k=request.top_k,
            conversation_history=request.conversation_history
        )

        return structured_response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(error)}"
        )
