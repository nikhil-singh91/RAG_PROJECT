"""
Retrieval module for the PDF RAG Application.
Performs semantic similarity search against the ChromaDB collection using
Sentence Transformer embeddings, calculating similarity scores and filtering.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional
from src.embeddings import EmbeddingManager
from src.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with relevance metrics and citations."""
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    distance: float
    similarity_score: float

    def format_citation(self) -> str:
        """Formatted citation string for the retrieved chunk."""
        return f"{self.document_name} (Page {self.page_number})"


@dataclass
class RetrievalResult:
    """Result container for query semantic search."""
    query: str
    chunks: List[RetrievedChunk]
    total_found: int
    has_results: bool
    status_message: str


class Retriever:
    """
    Coordinates semantic vector retrieval between query embedding and ChromaDB.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_manager: EmbeddingManager,
    ):
        """
        Initialize Retriever.

        Args:
            vector_db: Initialized VectorDatabase instance.
            embedding_manager: Initialized EmbeddingManager using the same model as indexing.
        """
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filter_doc_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve top_k most relevant chunks for a user query.

        Args:
            query: User's question or search query string.
            top_k: Maximum number of chunks to return.
            similarity_threshold: Optional minimum similarity score (0.0 to 1.0).
            filter_doc_id: Optional document_id filter if querying a specific document.

        Returns:
            RetrievalResult containing matched chunks, citations, and scores.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return RetrievalResult(
                query=query,
                chunks=[],
                total_found=0,
                has_results=False,
                status_message="Query is empty.",
            )

        # Check if database has any chunks
        total_chunks = self.vector_db.count_total_chunks()
        if total_chunks == 0:
            return RetrievalResult(
                query=clean_query,
                chunks=[],
                total_found=0,
                has_results=False,
                status_message="No documents have been indexed yet. Please upload a PDF first.",
            )

        # Generate query vector with the same embedding model
        try:
            query_vector = self.embedding_manager.embed_query(clean_query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return RetrievalResult(
                query=clean_query,
                chunks=[],
                total_found=0,
                has_results=False,
                status_message=f"Error embedding query: {str(e)}",
            )

        # Chroma query args
        where_clause = {"document_id": filter_doc_id} if filter_doc_id else None
        fetch_k = min(top_k, total_chunks)

        try:
            results = self.vector_db.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return RetrievalResult(
                query=clean_query,
                chunks=[],
                total_found=0,
                has_results=False,
                status_message=f"Vector database query failed: {str(e)}",
            )

        # Parse ChromaDB output
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks: List[RetrievedChunk] = []

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance d in [0, 2].
            # Cosine similarity s = 1.0 - d
            dist_float = float(dist) if dist is not None else 0.0
            sim_score = max(0.0, min(1.0, 1.0 - dist_float))

            # Apply similarity threshold cutoff if specified
            if similarity_threshold is not None and similarity_threshold > 0.0:
                if sim_score < similarity_threshold:
                    continue

            meta_dict: Dict[str, Any] = meta or {}
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    document_id=meta_dict.get("document_id", ""),
                    document_name=meta_dict.get("document_name", "Unknown Document"),
                    page_number=int(meta_dict.get("page_number", 1)),
                    chunk_index=int(meta_dict.get("chunk_index", 0)),
                    distance=round(dist_float, 4),
                    similarity_score=round(sim_score, 4),
                )
            )

        has_results = len(retrieved_chunks) > 0
        message = (
            f"Retrieved {len(retrieved_chunks)} relevant chunk(s)."
            if has_results
            else "No chunks met the similarity criteria for this query."
        )

        return RetrievalResult(
            query=clean_query,
            chunks=retrieved_chunks,
            total_found=len(retrieved_chunks),
            has_results=has_results,
            status_message=message,
        )
