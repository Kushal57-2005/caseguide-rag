from app.services.embedding_service import generate_embedding
from app.services.vector_store import search_chunks


def retrieve_documents(
    question: str,
    top_k: int = 5,
    min_similarity: float | None = None,
    document_id: str | None = None
):

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

    # Step 4: Process results
    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        # Cosine distance -> cosine similarity
        similarity = 1 - distance

        if (
            min_similarity is not None
            and similarity < min_similarity
        ):
            continue

        retrieved_chunks.append({
            "chunk_id": results["ids"][0][i],

            "content": results["documents"][0][i],

            "metadata": results["metadatas"][0][i],

            "similarity": similarity,

            "distance": distance
        })

    return retrieved_chunks