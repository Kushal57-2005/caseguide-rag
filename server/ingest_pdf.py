from pathlib import Path
from uuid import uuid4

from app.services.pdf_loader import extract_text_from_pdf
from app.services.text_cleaner import clean_text
from app.services.text_splitter import split_text
from app.services.embedding_service import generate_embedding
from app.services.vector_store import add_chunks


PDF_PATH = "storage/pdf/rain_data.pdf"

def ingest_pdf(pdf_path: str):
    document_id = str(uuid4())

    print("1. Extracting PDF text...")

    document = extract_text_from_pdf(pdf_path)

    all_chunks = []

    print("2. Cleaning and chunking...")

    for page in document["pages"]:
        cleaned_text = clean_text(page["text"])

        chunks = split_text(
            cleaned_text,
            chunk_size=1000,
            chunk_overlap=200
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "document_id": document_id,
                    "filename": document["filename"],
                    "page_number": page["page_number"],
                    "chunk_index": index
                }
            })

    print(f"Total chunks created: {len(all_chunks)}")

    print("3. Generating embeddings and storing...")

    batch_size = 32

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]

        embeddings = [
            generate_embedding(chunk["content"])
            for chunk in batch
        ]

        add_chunks(batch, embeddings)

        print(
            f"Stored {min(start + batch_size, len(all_chunks))}"
            f"/{len(all_chunks)} chunks"
        )

    print("\nPDF ingestion completed!")
    print("Document ID:", document_id)

    return document_id


if __name__ == "__main__":
    ingest_pdf(PDF_PATH)