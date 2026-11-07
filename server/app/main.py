from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from scalar_fastapi import get_scalar_api_reference

from app.api.routes import chat, documents

app = FastAPI(
    title="RAG API",
    description="PDF-based Retrieval Augmented Generation API",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat.router)
app.include_router(documents.router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
def serve_ui():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "RAG API is running"}


@app.get("/ui", include_in_schema=False)
def serve_ui_explicit():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "RAG API is running"}


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy"
    }


@app.get("/scalar", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="My API Documentation"
    )
