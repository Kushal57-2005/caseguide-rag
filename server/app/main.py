from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
from app.api.routes import chat, documents

app = FastAPI(
    title="RAG API",
    description="PDF-based Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
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
