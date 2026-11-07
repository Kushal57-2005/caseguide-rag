import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "PDF-based Retrieval-Augmented Generation API using NVIDIA NIM and ChromaDB"

    # Embedding Settings
    NVIDIA_EMBEDDING_API_KEY: str = os.getenv("NVIDIA_EMBEDDING_API_KEY", "")
    NVIDIA_EMBEDDING_BASE_URL: str = os.getenv("NVIDIA_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")

    # Generation (LLM) Settings
    NVIDIA_LLM_API_KEY: str = os.getenv("NVIDIA_LLM_API_KEY", "")
    NVIDIA_LLM_BASE_URL: str = os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "google/diffusiongemma-26b-a4b-it")

    # Storage Paths
    CHROMA_PATH: str = "./storage/chroma_db"
    PDF_DIR: str = "storage/pdf"


settings = Settings()

