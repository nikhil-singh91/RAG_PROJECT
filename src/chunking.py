"""
Text Chunking module for the PDF RAG Application.
Splits extracted document pages into semantically coherent chunks using
RecursiveCharacterTextSplitter while strictly preserving page-level source metadata.
"""

from dataclasses import dataclass
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.pdf_loader import PageData, PDFExtractionResult


@dataclass
class TextChunk:
    """Represents an individual text chunk with comprehensive source metadata."""
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    char_count: int


@dataclass
class ChunkingResult:
    """Summary of chunking process across one or multiple documents."""
    chunks: List[TextChunk]
    total_chunks: int
    total_characters: int
    document_id: str
    document_name: str


class DocumentChunker:
    """
    Chunks text using LangChain's RecursiveCharacterTextSplitter.
    Splits page-by-page so each chunk is directly attributable to a specific PDF page.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk (default: 1000).
            chunk_overlap: Overlapping characters between consecutive chunks (default: 200).
            separators: Hierarchy of split delimiters (paragraphs, newlines, spaces).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_page(self, page_data: PageData, start_chunk_idx: int = 0) -> List[TextChunk]:
        """
        Split a single page's text into chunks while preserving page metadata.

        Args:
            page_data: PageData object containing text and page metadata.
            start_chunk_idx: Global index offset for unique IDs.

        Returns:
            List of TextChunk objects.
        """
        if not page_data.text or page_data.is_empty:
            return []

        # Use RecursiveCharacterTextSplitter to split page text
        split_texts = self.splitter.split_text(page_data.text)
        chunks: List[TextChunk] = []

        for idx, text_segment in enumerate(split_texts):
            clean_segment = text_segment.strip()
            if not clean_segment:
                continue

            chunk_global_idx = start_chunk_idx + idx
            # Deterministic, unique ID per chunk
            chunk_id = f"{page_data.document_id}_p{page_data.page_number}_c{chunk_global_idx}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=clean_segment,
                    document_id=page_data.document_id,
                    document_name=page_data.document_name,
                    page_number=page_data.page_number,
                    chunk_index=chunk_global_idx,
                    char_count=len(clean_segment),
                )
            )

        return chunks

    def chunk_extraction_result(
        self, extraction_result: PDFExtractionResult
    ) -> ChunkingResult:
        """
        Chunk an entire extracted PDF document page by page.

        Args:
            extraction_result: Output from PDFLoader.load_pdf.

        Returns:
            ChunkingResult with all generated chunks and summary statistics.
        """
        all_chunks: List[TextChunk] = []
        total_chars = 0
        chunk_counter = 0

        for page in extraction_result.extracted_pages:
            if page.is_empty or not page.text:
                continue

            page_chunks = self.chunk_page(page, start_chunk_idx=chunk_counter)
            for c in page_chunks:
                all_chunks.append(c)
                total_chars += c.char_count
                chunk_counter += 1

        return ChunkingResult(
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            total_characters=total_chars,
            document_id=extraction_result.document_id,
            document_name=extraction_result.document_name,
        )
