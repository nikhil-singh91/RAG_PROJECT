"""
Vector Database module for the PDF RAG Application.
Manages persistent storage, metadata indexing, duplicate detection, and retrieval
operations using ChromaDB.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from src.chunking import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class StoredDocumentInfo:
    """Summary of a document stored within the ChromaDB collection."""
    document_id: str
    document_name: str
    chunk_count: int
    page_count: int
    pages_indexed: List[int]


class VectorDatabase:
    """
    Encapsulates ChromaDB persistent storage for RAG chunk embeddings and metadata.
    """

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        collection_name: str = "pdf_rag_collection",
    ):
        """
        Initialize persistent ChromaDB client and collection.

        Args:
            persist_directory: Local filesystem directory for database files.
            collection_name: Unique collection name.
        """
        self.persist_directory = str(Path(persist_directory).resolve())
        self.collection_name = collection_name

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized at '{self.persist_directory}' with collection '{self.collection_name}'."
        )

    def is_document_indexed(self, document_id: str) -> bool:
        """
        Check if chunks corresponding to document_id already exist in the collection.
        Prevents duplicate insertion of the same PDF.
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1,
            )
            return bool(results and results.get("ids") and len(results["ids"]) > 0)
        except Exception as e:
            logger.warning(f"Error checking document existence for '{document_id}': {e}")
            return False

    def get_document_chunk_count(self, document_id: str) -> int:
        """Get the number of chunks stored for a specific document."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
            )
            return len(results.get("ids", [])) if results else 0
        except Exception:
            return 0

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Insert chunks with their embeddings and metadata into ChromaDB.
        Batches operations to prevent memory spikes.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Corresponding list of embedding vectors.
            batch_size: Batch size for ChromaDB insertion.

        Returns:
            Number of successfully inserted chunks.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        # Filter out if document is already indexed to prevent duplicates
        doc_id = chunks[0].document_id
        if self.is_document_indexed(doc_id):
            logger.info(f"Document '{doc_id}' is already indexed. Skipping re-insertion.")
            return self.get_document_chunk_count(doc_id)

        total_inserted = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [chunk.chunk_id for chunk in batch_chunks]
            documents = [chunk.text for chunk in batch_chunks]
            metadatas: List[Dict[str, Any]] = [
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": int(chunk.page_number),
                    "chunk_index": int(chunk.chunk_index),
                    "char_count": int(chunk.char_count),
                }
                for chunk in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )
            total_inserted += len(batch_chunks)

        logger.info(f"Successfully added {total_inserted} chunks to ChromaDB.")
        return total_inserted

    def get_all_stored_documents(self) -> List[StoredDocumentInfo]:
        """
        Retrieve summary information of all unique documents currently stored.
        """
        try:
            # Query all metadata without pulling all text/embeddings
            data = self.collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", []) or []

            doc_map: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                d_id = meta.get("document_id")
                d_name = meta.get("document_name", "Unknown Document")
                p_num = meta.get("page_number", 1)

                if d_id not in doc_map:
                    doc_map[d_id] = {
                        "document_id": d_id,
                        "document_name": d_name,
                        "chunk_count": 0,
                        "pages": set(),
                    }

                doc_map[d_id]["chunk_count"] += 1
                doc_map[d_id]["pages"].add(p_num)

            summaries: List[StoredDocumentInfo] = []
            for d_id, info in doc_map.items():
                sorted_pages = sorted(list(info["pages"]))
                summaries.append(
                    StoredDocumentInfo(
                        document_id=d_id,
                        document_name=info["document_name"],
                        chunk_count=info["chunk_count"],
                        page_count=len(sorted_pages),
                        pages_indexed=sorted_pages,
                    )
                )

            return summaries
        except Exception as e:
            logger.error(f"Failed to list stored documents: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks and vectors associated with a specific document ID.

        Args:
            document_id: Unique document ID to remove.

        Returns:
            True if deletion was executed, False on error.
        """
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted all vectors for document_id: '{document_id}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{document_id}': {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all entries in the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{self.collection_name}' has been cleared.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def count_total_chunks(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
