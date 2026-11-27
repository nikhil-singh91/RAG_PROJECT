"""
PDF Loader module for the PDF RAG Application.
Extracts text page-by-page using PyMuPDF, preserves metadata,
detects scanned documents, and handles corrupted/encrypted PDFs gracefully.
"""

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import List, Optional, Union
import pymupdf as fitz


@dataclass
class PageData:
    """Represents text and metadata extracted from a single PDF page."""
    page_number: int  # 1-based indexing for user-friendly citations
    text: str
    document_name: str
    document_id: str
    char_count: int
    is_empty: bool = False


@dataclass
class PDFExtractionResult:
    """Represents the complete result of a PDF document extraction."""
    success: bool
    document_name: str
    document_id: str
    total_pages: int
    extracted_pages: List[PageData]
    total_characters: int
    is_scanned: bool = False
    warning_message: Optional[str] = None
    error_message: Optional[str] = None


def generate_document_id(content_bytes: bytes, filename: str) -> str:
    """
    Generate a stable, unique document ID based on content SHA256 and filename.
    Ensures identical documents have the same ID to prevent redundant vector storage.
    """
    hasher = hashlib.sha256()
    hasher.update(content_bytes)
    return f"{hasher.hexdigest()[:16]}_{Path(filename).stem}"


class PDFLoader:
    """
    Extracts text page-by-page from PDF files using PyMuPDF.
    Handles scanned documents, password-protected files, and empty pages.
    """

    def __init__(self, min_char_threshold_for_scanned: int = 40):
        """
        Args:
            min_char_threshold_for_scanned: Average characters per page below which
                                           a multi-page document is flagged as scanned.
        """
        self.min_char_threshold = min_char_threshold_for_scanned

    def load_pdf(
        self,
        file_source: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        save_copy_path: Optional[Path] = None,
    ) -> PDFExtractionResult:
        """
        Extract text page by page from a file path, raw bytes, or BytesIO object.

        Args:
            file_source: File path (str/Path) or in-memory bytes/BytesIO.
            filename: Original name of the document.
            save_copy_path: Optional path to persist a copy of the uploaded file.

        Returns:
            PDFExtractionResult with extracted pages, metadata, and status warnings.
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            doc_name = filename or path.name
            if not path.exists():
                return PDFExtractionResult(
                    success=False,
                    document_name=doc_name,
                    document_id="",
                    total_pages=0,
                    extracted_pages=[],
                    total_characters=0,
                    error_message=f"File not found at path: {path}",
                )
            try:
                pdf_bytes = path.read_bytes()
            except Exception as e:
                return PDFExtractionResult(
                    success=False,
                    document_name=doc_name,
                    document_id="",
                    total_pages=0,
                    extracted_pages=[],
                    total_characters=0,
                    error_message=f"Failed to read file: {str(e)}",
                )
        elif isinstance(file_source, io.BytesIO):
            doc_name = filename or "document.pdf"
            pdf_bytes = file_source.getvalue()
        elif isinstance(file_source, bytes):
            doc_name = filename or "document.pdf"
            pdf_bytes = file_source
        else:
            return PDFExtractionResult(
                success=False,
                document_name=filename or "unknown.pdf",
                document_id="",
                total_pages=0,
                extracted_pages=[],
                total_characters=0,
                error_message="Unsupported file source type provided to PDFLoader.",
            )

        # Generate unique document ID
        doc_id = generate_document_id(pdf_bytes, doc_name)

        # Optionally persist copy to data directory
        if save_copy_path:
            try:
                save_copy_path.parent.mkdir(parents=True, exist_ok=True)
                save_copy_path.write_bytes(pdf_bytes)
            except Exception:
                pass

        # Open PDF with PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            return PDFExtractionResult(
                success=False,
                document_name=doc_name,
                document_id=doc_id,
                total_pages=0,
                extracted_pages=[],
                total_characters=0,
                error_message=f"Corrupted or invalid PDF format: {str(e)}",
            )
        except Exception as e:
            return PDFExtractionResult(
                success=False,
                document_name=doc_name,
                document_id=doc_id,
                total_pages=0,
                extracted_pages=[],
                total_characters=0,
                error_message=f"Failed to open PDF document: {str(e)}",
            )

        # Check for password encryption
        if doc.is_encrypted:
            doc.close()
            return PDFExtractionResult(
                success=False,
                document_name=doc_name,
                document_id=doc_id,
                total_pages=0,
                extracted_pages=[],
                total_characters=0,
                error_message=(
                    "This PDF is password-protected or encrypted. "
                    "Please decrypt the file before uploading."
                ),
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return PDFExtractionResult(
                success=False,
                document_name=doc_name,
                document_id=doc_id,
                total_pages=0,
                extracted_pages=[],
                total_characters=0,
                error_message="PDF contains no pages.",
            )

        extracted_pages: List[PageData] = []
        total_chars = 0
        pages_with_images = 0

        # Extract page by page preserving 1-based page index
        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                cleaned_text = raw_text.strip()
                char_count = len(cleaned_text)
                is_empty = (char_count == 0)

                if len(page.get_images()) > 0:
                    pages_with_images += 1

                extracted_pages.append(
                    PageData(
                        page_number=page_num,
                        text=cleaned_text,
                        document_name=doc_name,
                        document_id=doc_id,
                        char_count=char_count,
                        is_empty=is_empty,
                    )
                )
                total_chars += char_count

            except Exception:
                extracted_pages.append(
                    PageData(
                        page_number=page_num,
                        text="",
                        document_name=doc_name,
                        document_id=doc_id,
                        char_count=0,
                        is_empty=True,
                    )
                )

        doc.close()

        # Scanned PDF Detection
        avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
        is_scanned = (avg_chars_per_page < self.min_char_threshold) and (total_pages > 0)

        warning_message = None
        if is_scanned:
            warning_message = (
                f"Warning: '{doc_name}' appears to be a scanned document or image-only PDF "
                f"(extracted average of only {avg_chars_per_page:.1f} characters/page). "
                "PyMuPDF extracts embedded selectable text. To index scanned images, "
                "an OCR tool (such as Tesseract or Google Cloud Vision) is required."
            )

        return PDFExtractionResult(
            success=True,
            document_name=doc_name,
            document_id=doc_id,
            total_pages=total_pages,
            extracted_pages=extracted_pages,
            total_characters=total_chars,
            is_scanned=is_scanned,
            warning_message=warning_message,
        )
