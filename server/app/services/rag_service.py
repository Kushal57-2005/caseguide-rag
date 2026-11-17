import logging
from typing import Union, List, Optional
from app.schemas.chat_schema import (
    ChatRequest,
    StructuredRAGResponse,
    FallbackResponse,
    FALLBACK_MESSAGE
)
from app.services.retrieval_service import retrieve_documents, SIMILARITY_THRESHOLD
from app.services.query_understanding import understand_query
from app.services.generation_service import generate_structured_response

logger = logging.getLogger("caseguide.rag_service")
logging.basicConfig(level=logging.INFO)


def execute_rag_pipeline(
    request: ChatRequest,
    min_similarity: Optional[float] = None
) -> Union[StructuredRAGResponse, FallbackResponse]:
    """
    Executes the CaseGuide RAG pipeline strictly according to the Grounded Response Policy:

    1. Performs vector database retrieval using the user query.
    2. Calculates cosine similarity scores for all retrieved chunks.
    3. Applies configurable similarity threshold (default: 0.40).
    4. Removes all documents below the threshold.
    5. FALLBACK CHECK (IMMEDIATELY AFTER FILTERING, BEFORE ANY LLM CALL):
       If no relevant documents remain:
       - DO NOT call the LLM.
       - DO NOT call the query refinement LLM.
       - DO NOT generate a structured response.
       - Directly return the backend predefined fallback response.
    6. SUCCESSFUL RETRIEVAL:
       If at least one sufficiently relevant document (>= 0.40) exists:
       - Continue existing RAG pipeline (query refinement, generation, citations).
       - Pass ONLY the filtered documents to the LLM.
    """
    threshold = min_similarity if min_similarity is not None else SIMILARITY_THRESHOLD

    # Step 1 & 2 & 3 & 4: Dense Vector Retrieval + Similarity Calculation + Filtering
    # Note: Only embedding API is invoked here for vector search; NO LLM is called.
    retrieved_chunks = retrieve_documents(
        question=request.question,
        top_k=request.top_k,
        min_similarity=threshold,
        document_id=request.document_id
    )

    # Step 5: Fallback Gate - Immediately evaluated BEFORE any LLM invocation
    if not retrieved_chunks:
        logger.warning(
            f"[FALLBACK TRIGGERED] Query: '{request.question}' | "
            f"Retrieved chunks >= threshold ({threshold}): 0 | "
            f"LLM calls: 0. Returning backend fallback response directly."
        )
        return FallbackResponse(
            status="fallback",
            response=FALLBACK_MESSAGE,
            sources=[]
        )

    # Step 6: Relevant documents verified (>= 0.40 similarity) -> Proceed with RAG Pipeline
    logger.info(
        f"[RELEVANT DOCS FOUND] Query: '{request.question}' | "
        f"Verified chunks >= {threshold}: {len(retrieved_chunks)} | "
        f"Proceeding with LLM query understanding and answer generation."
    )

    # Invoke Query Understanding only for valid, relevant queries
    query_understanding = understand_query(
        original_query=request.question,
        conversation_history=request.conversation_history
    )

    # Generate grounded structured response passing ONLY filtered chunks
    structured_response = generate_structured_response(
        original_query=request.question,
        query_understanding=query_understanding,
        retrieved_chunks=retrieved_chunks,
        top_k=request.top_k,
        conversation_history=request.conversation_history
    )

    return structured_response
