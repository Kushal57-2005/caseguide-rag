from pathlib import Path
from uuid import uuid4

from app.services.pdf_loader import extract_text_from_pdf
from app.services.text_cleaner import clean_text
from app.services.text_splitter import split_text
from app.services.embedding_service import generate_embedding
from app.services.vector_store import add_chunks


def ingest_pdf_file(
    file_path: str,
    original_filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32
) -> dict:
    document_id = str(uuid4())
    display_filename = original_filename or Path(file_path).name

    # 1. Extract text from PDF
    document = extract_text_from_pdf(file_path)

    all_chunks = []

    # 2. Clean and chunk
    for page in document["pages"]:
        cleaned_text = clean_text(page["text"])

        chunks = split_text(
            cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "document_id": document_id,
                    "filename": display_filename,
                    "page_number": page["page_number"],
                    "chunk_index": index
                }
            })

    # 3. Generate embeddings and store in ChromaDB
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]

        embeddings = [
            generate_embedding(chunk["content"], input_type="passage")
            for chunk in batch
        ]

        add_chunks(batch, embeddings)

    return {
        "document_id": document_id,
        "filename": display_filename,
        "total_pages": document["total_pages"],
        "total_chunks": len(all_chunks)
    }

