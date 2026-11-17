import os
from typing import List, Optional
from app.services.embedding_service import generate_embedding
from app.services.vector_store import search_chunks

# Default similarity threshold: chunks with similarity < 0.4 are considered irrelevant
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))


def retrieve_documents(
    question: str,
    top_k: int = 5,
    min_similarity: float | None = SIMILARITY_THRESHOLD,
    document_id: str | None = None
) -> List[dict]:
    """
    Retrieves top_k chunks from ChromaDB for the given query.
    Calculates cosine similarity correctly from ChromaDB's cosine distance metric.
    Filters out any chunks with similarity_score < min_similarity (default: 0.4).
    Preserves all retrieval metadata (chunk_id, document_id, file_name, page_number, raw_distance, similarity_score, retrieval_rank).
    """

    # Step 1: Generate query embedding
    query_embedding = generate_embedding(question, input_type="query")

    # Step 2: Prepare metadata filter
    where = None
    if document_id:
        where = {
            "document_id": document_id
        }

    # Step 3: Search ChromaDB
    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        where=where
    )

    retrieved_chunks = []
    ids = results["ids"][0] if results.get("ids") and len(results["ids"]) > 0 else []
    docs = results["documents"][0] if results.get("documents") and len(results["documents"]) > 0 else []
    metas = results["metadatas"][0] if results.get("metadatas") and len(results.get("metadatas")) > 0 else []
    dists = results["distances"][0] if results.get("distances") and len(results.get("distances")) > 0 else []

    # Step 4: Process and strictly format results
    for i in range(len(ids)):
        raw_dist = float(dists[i]) if i < len(dists) else 0.0
        
        # In ChromaDB with hnsw:space = "cosine", distance d = 1 - cosine_similarity
        # Hence similarity_score = 1.0 - raw_distance
        similarity = round(max(-1.0, min(1.0, 1.0 - raw_dist)), 4)

        if min_similarity is not None and similarity < min_similarity:
            continue

        meta = metas[i] if i < len(metas) and metas[i] else {}
        doc_id = str(meta.get("document_id", "unknown"))
        file_name = str(meta.get("filename", "unknown"))
        page_num = int(meta.get("page_number", 1))
        content = docs[i] if i < len(docs) else ""

        retrieved_chunks.append({
            "chunk_id": ids[i],
            "document_id": doc_id,
            "file_name": file_name,
            "page_number": page_num,
            "raw_distance": round(raw_dist, 4),
            "similarity_score": similarity,
            "reranker_score": None,
            "retrieval_rank": len(retrieved_chunks) + 1,
            "content": content,
            "metadata": meta
        })

    return retrieved_chunks
