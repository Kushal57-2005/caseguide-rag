import chromadb

client = chromadb.PersistentClient(
    path="./storage/chroma_db"
)

COLLECTION_NAME = "pdf_documents"

# IMPORTANT:
# Delete the old collection only once during development
# if it contains 3-dimensional dummy embeddings.

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


def add_chunks(chunks, embeddings):

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        chunk_id = (
            f"{metadata['document_id']}_"
            f"{metadata['page_number']}_"
            f"{metadata['chunk_index']}"
        )

        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def search_chunks(
    query_embedding,
    top_k=5,
    where=None
):

    total_count = collection.count()

    if total_count == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, total_count),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if where:
        search_params["where"] = where

    return collection.query(**search_params)


def delete_document(document_id: str):

    collection.delete(
        where={
            "document_id": document_id
        }
    )


def list_documents() -> list[dict]:
    data = collection.get(include=["metadatas"])
    docs = {}
    for m in data.get("metadatas", []):
        if not m:
            continue
        doc_id = m.get("document_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "filename": m.get("filename", "unknown"),
                "total_chunks": 0,
                "pages": set()
            }
        docs[doc_id]["total_chunks"] += 1
        page = m.get("page_number")
        if page is not None:
            docs[doc_id]["pages"].add(page)

    return [
        {
            "document_id": d["document_id"],
            "filename": d["filename"],
            "total_chunks": d["total_chunks"],
            "total_pages": len(d["pages"])
        }
        for d in docs.values()
    ]
