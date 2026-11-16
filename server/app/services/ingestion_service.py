from pathlib import Path
from uuid import uuid4

from app.services.pdf_loader import extract_text_from_pdf
from app.services.text_cleaner import clean_text
from app.services.text_splitter import split_text
from app.services.embedding_service import generate_embedding, generate_embeddings
from app.services.vector_store import add_chunks


def ingest_pdf_file(
    file_path: str,
    original_filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    show_progress: bool = True
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

    if show_progress:
        print(f"  -> Extracted {len(all_chunks)} chunks across {document['total_pages']} pages. Indexing embeddings...", flush=True)

    # 3. Generate batch embeddings and store in ChromaDB
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        texts = [chunk["content"] for chunk in batch]

        embeddings = generate_embeddings(texts, input_type="passage")

        add_chunks(batch, embeddings)

        if show_progress:
            done = min(start + batch_size, len(all_chunks))
            pct = int((done / len(all_chunks)) * 100)
            print(f"\r  [{pct:3d}%] Indexed {done}/{len(all_chunks)} chunks...", end="", flush=True)

    if show_progress and all_chunks:
        print(flush=True)

    return {
        "document_id": document_id,
        "filename": display_filename,
        "total_pages": document["total_pages"],
        "total_chunks": len(all_chunks)
    }


def ingest_folder(
    folder_path: str,
    recursive: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32
) -> dict:
    """
    Ingests all PDF files found in a specified folder into ChromaDB.
    """
    path = Path(folder_path)

    if not path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory.")

    pdf_files = list(path.rglob("*.pdf") if recursive else path.glob("*.pdf"))

    if not pdf_files:
        return {
            "folder": str(path),
            "total_files_found": 0,
            "processed_documents": [],
            "total_chunks": 0
        }

    processed = []
    total_chunks = 0
    errors = []

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing '{pdf_file.name}'...", flush=True)
        try:
            result = ingest_pdf_file(
                file_path=str(pdf_file),
                original_filename=pdf_file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                show_progress=True
            )
            processed.append(result)
            total_chunks += result["total_chunks"]
            print(f"  ✓ Completed '{pdf_file.name}' ({result['total_chunks']} chunks stored)", flush=True)
        except Exception as e:
            print(f"  ✗ Error in '{pdf_file.name}': {e}", flush=True)
            errors.append({
                "filename": pdf_file.name,
                "error": str(e)
            })

    return {
        "folder": str(path),
        "total_files_found": len(pdf_files),
        "successful_ingestions": len(processed),
        "processed_documents": processed,
        "total_chunks": total_chunks,
        "errors": errors
    }


