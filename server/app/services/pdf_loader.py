import pymupdf as fitz
def extract_text_from_pdf(file_path: str):
    document = fitz.open(file_path)
    pages=[]

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
                "page_number": page_number,
                "text": text
        })

    result = {
        "filename": document.name,
        "total_pages": len(document),
        "pages": pages
    }

    document.close()

    return result