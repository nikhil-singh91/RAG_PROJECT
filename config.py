"""
Configuration module for the PDF RAG Application.
Loads settings from environment variables and defines application constants.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base project directory
BASE_DIR: Path = Path(__file__).resolve().parent

# Persistent Storage Directories
DATA_DIR: Path = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR: Path = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Central configuration class for PDF RAG pipeline."""

    # Gemini LLM Settings
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip()
    )
    default_gemini_model: str = "gemini-2.5-flash"
    available_gemini_models: List[str] = field(
        default_factory=lambda: [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]
    )
    llm_temperature: float = 0.2

    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""]
    )

    # Embedding Settings (Local Sentence Transformers)
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    available_embedding_models: List[str] = field(
        default_factory=lambda: [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        ]
    )

    # ChromaDB Vector Storage Settings
    chroma_collection_name: str = "pdf_rag_collection"
    chroma_persist_dir: str = str(VECTOR_DB_DIR)

    # Retrieval Settings
    default_top_k: int = 5
    default_similarity_threshold: float = 0.0  # 0.0 means return all top_k without cutoff

    def is_gemini_key_set(self) -> bool:
        """Check if GEMINI_API_KEY is configured."""
        return bool(self.gemini_api_key and not self.gemini_api_key.startswith("your_"))


# Global configuration instance
config = AppConfig()
