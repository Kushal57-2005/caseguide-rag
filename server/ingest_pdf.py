import sys
from pathlib import Path

from app.services.ingestion_service import ingest_pdf_file, ingest_folder

DEFAULT_PATH = "storage/pdf"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    target_path = Path(target)

    if not target_path.exists():
        print(f"Error: Path '{target}' does not exist.")
        sys.exit(1)

    if target_path.is_dir():
        print(f"\nScanning folder: {target_path.resolve()} for PDF files...")
        summary = ingest_folder(str(target_path))

        print(f"\n================ FOLDER INGESTION SUMMARY ================")
        print(f"Folder: {summary['folder']}")
        print(f"PDF files found: {summary['total_files_found']}")
        print(f"Successfully ingested: {summary['successful_ingestions']}")
        print(f"Total chunks created: {summary['total_chunks']}")

        if summary["processed_documents"]:
            print("\nIngested Documents:")
            for doc in summary["processed_documents"]:
                print(f" - {doc['filename']} | Pages: {doc['total_pages']} | Chunks: {doc['total_chunks']} | ID: {doc['document_id']}")

        if summary["errors"]:
            print("\nErrors encountered:")
            for err in summary["errors"]:
                print(f" - {err['filename']}: {err['error']}")

        print("=========================================================\n")

    elif target_path.is_file() and target_path.suffix.lower() == ".pdf":
        print(f"\nIngesting single PDF: {target_path.resolve()}...")
        result = ingest_pdf_file(str(target_path), original_filename=target_path.name)
        print(f"\n================ INGESTION SUCCESS ================")
        print(f"Document ID: {result['document_id']}")
        print(f"Filename: {result['filename']}")
        print(f"Pages: {result['total_pages']}")
        print(f"Total Chunks: {result['total_chunks']}")
        print("===================================================\n")
    else:
        print(f"Error: '{target}' is not a PDF file or a folder.")
        sys.exit(1)


if __name__ == "__main__":
    main()
