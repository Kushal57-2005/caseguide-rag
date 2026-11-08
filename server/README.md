# PDF-Based RAG API

A high-performance Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **ChromaDB**, and **NVIDIA NIM** (Nemotron embeddings & LLM generation).

---

## Architecture Overview

```
[ Client / Frontend ]
          │
          ▼
┌──────────────────────────────────────────────┐
│                  FastAPI                     │
│  ├── POST /api/chat/                         │
│  ├── POST /api/documents/upload              │
│  ├── GET  /api/documents/                    │
│  └── DELETE /api/documents/{document_id}     │
└───────┬───────────────────────────────┬──────┘
        │                               │
        ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   Retrieval Service     │   │   Ingestion Service     │
│  ├── Query Embedding    │   │  ├── PyMuPDF Text Extract│
│  │   (Nemotron)         │   │  ├── Text Cleaning      │
│  └── ChromaDB Search    │   │  ├── Text Chunking      │
└──────────┬──────────────┘   │  └── Passage Embedding  │
           │                  └───────────┬─────────────┘
           │ (Context Chunks)             │
           ▼                              ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   Generation Service    │   │       ChromaDB          │
│  ├── System Prompt      │   │  (Persistent Vector DB) │
│  └── Nemotron / Gemma   │   └─────────────────────────┘
└─────────────────────────┘
```

---

## Features

- **Document Ingestion**: Upload PDF documents via REST API. Text is extracted, cleaned, chunked with overlap, and indexed.
- **Semantic Vector Search**: Uses NVIDIA's `nemotron-3.5-embed-1b` model for dense vector embeddings stored in a persistent ChromaDB instance.
- **Strict Grounded Generation**: Context chunks and user queries are passed with strict grounding rules to ensure answers cite sources and prevent hallucinations.
- **Document Management**: Endpoints to list indexed documents, inspect page/chunk counts, and remove documents.
- **Interactive Documentation**: Available through both Swagger UI (`/docs`) and Scalar UI (`/scalar`).

---

## Getting Started

### 1. Set up virtual environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your NVIDIA API credentials:

```env
# Embedding Service
NVIDIA_EMBEDDING_API_KEY=nvapi-...
NVIDIA_EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-1b

# Generation Service
NVIDIA_LLM_API_KEY=nvapi-...
NVIDIA_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=google/diffusiongemma-26b-a4b-it
```

> Obtain API keys from [build.nvidia.com](https://build.nvidia.com/).

---

## Running the Server

Start the API server with live reloading:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

- **Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Scalar Documentation**: [http://127.0.0.1:8000/scalar](http://127.0.0.1:8000/scalar)

---

## API Reference

### 1. Chat & Query (`POST /api/chat/`)

Query your indexed documents:

```bash
curl -X POST "http://127.0.0.1:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG and how does it work?",
    "top_k": 5
  }'
```

### 2. Upload Document (`POST /api/documents/upload`)

Upload and index a PDF file:

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/upload" \
  -F "file=@/path/to/document.pdf"
```

### 3. List Documents (`GET /api/documents/`)

List all documents indexed in the vector store:

```bash
curl -X GET "http://127.0.0.1:8000/api/documents/"
```

### 4. Ingest an Entire Folder (`POST /api/documents/ingest-folder`)

Batch ingest all PDF files in a server directory:

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/ingest-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "storage/pdf",
    "recursive": false
  }'
```

### 5. Delete Document (`DELETE /api/documents/{document_id}`)

Remove a document and all its chunks from the vector store:

```bash
curl -X DELETE "http://127.0.0.1:8000/api/documents/{document_id}"
```

