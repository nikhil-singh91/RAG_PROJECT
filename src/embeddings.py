"""
Local Embedding Generation module for the PDF RAG Application.
Uses Sentence Transformers to generate local numerical dense vector embeddings
without relying on paid third-party embedding APIs.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages Sentence Transformer embedding models for generating local embeddings.
    Caches loaded models to optimize memory and speed.
    """

    _instances = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: Computing device ('cpu', 'cuda', 'mps', or None for auto-detect).
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load and cache the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

            # Compatible embedding dimension discovery
            dim_fn = getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", lambda: 384),
            )
            self.embedding_dimension = dim_fn()
            logger.info(
                f"Successfully loaded '{self.model_name}' (dimension: {self.embedding_dimension})"
            )
        except Exception as e:
            logger.error(f"Error loading embedding model '{self.model_name}': {e}")
            raise RuntimeError(
                f"Failed to initialize SentenceTransformer model '{self.model_name}': {str(e)}"
            )

    @classmethod
    def get_instance(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> "EmbeddingManager":
        """Singleton accessor to avoid re-instantiating the same model across app turns."""
        key = (model_name, device)
        if key not in cls._instances:
            cls._instances[key] = cls(model_name=model_name, device=device)
        return cls._instances[key]

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> List[List[float]]:
        """
        Convert a list of text strings into numerical embedding vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Mini-batch size for processing.
            show_progress_bar: Display progress bar in terminal/console.
            normalize_embeddings: Whether to L2-normalize vectors (recommended for cosine similarity).

        Returns:
            List of embedding vectors as Python float lists.
        """
        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

            # Convert numpy array to python nested lists for ChromaDB compatibility
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return [list(vec) for vec in embeddings]

        except Exception as e:
            logger.error(f"Error generating embeddings for {len(texts)} texts: {e}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def embed_query(
        self, query: str, normalize_embeddings: bool = True
    ) -> List[float]:
        """
        Convert a single query string into an embedding vector.

        Args:
            query: User's question or search query.
            normalize_embeddings: Whether to L2-normalize the vector.

        Returns:
            Embedding vector as a list of floats.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        try:
            embedding = self._model.encode(
                query.strip(),
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            return list(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for query '{query}': {e}")
            raise RuntimeError(f"Query embedding generation failed: {str(e)}")
