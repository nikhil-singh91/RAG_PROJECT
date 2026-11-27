"""
Comprehensive automated test suite for the PDF RAG application pipeline.
Tests text extraction, chunking, local embeddings, persistent ChromaDB,
deduplication, semantic search, document deletion, and error handling.
"""

import os
import sys
from pathlib import Path
import pymupdf as fitz

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import config
from src.pdf_loader import PDFLoader
from src.chunking import DocumentChunker
from src.embeddings import EmbeddingManager
from src.vector_db import VectorDatabase
from src.retrieval import Retriever
from src.llm import GeminiRAGGenerator


def create_test_pdf(pdf_path: Path) -> Path:
    """Creates a sample multi-page PDF document for testing."""
    doc = fitz.open()

    # Page 1: Overview
    page1 = doc.new_page()
    text1 = (
        "Introduction to Quantum Computing.\n\n"
        "Quantum computing is an emerging field that leverages the fundamental laws of quantum mechanics "
        "to solve problems beyond the capability of classical computers. While classical computers encode "
        "information in binary bits that can either be 0 or 1, quantum systems utilize quantum bits or qubits.\n\n"
        "This project explores quantum computing applications in cryptography, material science, and optimization."
    )
    page1.insert_text((50, 72), text1, fontsize=11)

    # Page 2: Principles
    page2 = doc.new_page()
    text2 = (
        "Quantum Superposition and Entanglement.\n\n"
        "The core advantage of quantum computation lies in two foundational phenomena: superposition and entanglement.\n\n"
        "Superposition allows a qubit to represent linear combinations of |0> and |1> states simultaneously. "
        "Because of this, an n-qubit quantum register can evaluate 2^n states concurrently.\n\n"
        "Quantum Entanglement links pairs or groups of particles such that the quantum state of each particle "
        "cannot be described independently of the state of the others, even when separated by large distances."
    )
    page2.insert_text((50, 72), text2, fontsize=11)

    # Page 3: Algorithms
    page3 = doc.new_page()
    text3 = (
        "Quantum Algorithms: Shor and Grover.\n\n"
        "Shor's Algorithm, formulated by Peter Shor in 1994, polynomial-time factors large integers, threatening RSA encryption.\n\n"
        "Grover's Algorithm, developed by Lov Grover in 1996, provides a quadratic speedup for searching unsorted databases, "
        "reducing search complexity from O(N) to O(sqrt(N)).\n\n"
        "Fault-tolerant quantum computation requires error mitigation techniques such as surface codes."
    )
    page3.insert_text((50, 72), text3, fontsize=11)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def run_all_tests():
    print("=" * 60)
    print("🚀 STARTING COMPLETE PDF RAG PIPELINE VERIFICATION")
    print("=" * 60)

    # Test 1: Module Imports
    print("\n[Test 1] Testing Module Imports...")
    assert PDFLoader is not None
    assert DocumentChunker is not None
    assert EmbeddingManager is not None
    assert VectorDatabase is not None
    assert Retriever is not None
    assert GeminiRAGGenerator is not None
    print("✅ All modules imported successfully.")

    # Test 2: Sample PDF Creation & Extraction
    print("\n[Test 2] Testing PDF Text Extraction with PyMuPDF...")
    test_pdf_path = BASE_DIR / "data" / "uploads" / "quantum_test.pdf"
    create_test_pdf(test_pdf_path)
    assert test_pdf_path.exists(), "Test PDF file was not created."

    loader = PDFLoader()
    result = loader.load_pdf(test_pdf_path)
    assert result.success is True, f"Extraction failed: {result.error_message}"
    assert result.total_pages == 3, f"Expected 3 pages, got {result.total_pages}"
    assert result.total_characters > 500, f"Characters count too low: {result.total_characters}"
    assert result.extracted_pages[0].page_number == 1, "Page 1 indexing incorrect."
    assert result.extracted_pages[1].page_number == 2, "Page 2 indexing incorrect."
    assert result.extracted_pages[2].page_number == 3, "Page 3 indexing incorrect."
    assert "Shor" in result.extracted_pages[2].text, "Expected Shor algorithm text on page 3."
    print(f"✅ Extracted {result.total_pages} pages, {result.total_characters} characters from '{result.document_name}'.")

    # Test 3: Text Chunking
    print("\n[Test 3] Testing Text Chunking & Metadata Preservation...")
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=100)
    chunk_res = chunker.chunk_extraction_result(result)
    assert chunk_res.total_chunks > 0, "No chunks generated."
    for c in chunk_res.chunks:
        assert c.document_id == result.document_id
        assert c.document_name == result.document_name
        assert c.page_number in [1, 2, 3]
        assert len(c.text) > 0
        assert c.chunk_id.startswith(result.document_id)
    print(f"✅ Generated {chunk_res.total_chunks} chunks with preserved page metadata.")

    # Test 4: Embedding Generation (Local Sentence Transformers)
    print("\n[Test 4] Testing Local Embedding Generation...")
    embed_mgr = EmbeddingManager.get_instance(config.default_embedding_model)
    chunk_texts = [c.text for c in chunk_res.chunks]
    embeddings = embed_mgr.embed_texts(chunk_texts, batch_size=8)
    assert len(embeddings) == len(chunk_texts), "Embeddings count does not match chunks count."
    assert len(embeddings[0]) == 384, f"Expected 384 dimensions, got {len(embeddings[0])}"
    
    # Test query embedding
    query_vec = embed_mgr.embed_query("What is Grover's algorithm?")
    assert len(query_vec) == 384, "Query vector dimension mismatch."
    print(f"✅ Generated {len(embeddings)} local 384-dimensional embeddings.")

    # Test 5: ChromaDB Persistent Storage
    print("\n[Test 5] Testing ChromaDB Persistent Vector Storage...")
    test_db_dir = BASE_DIR / "vectorstore_test"
    vdb = VectorDatabase(persist_directory=str(test_db_dir), collection_name="test_collection")
    vdb.clear_collection()

    inserted = vdb.add_chunks(chunk_res.chunks, embeddings)
    assert inserted == len(chunk_res.chunks), f"Expected {len(chunk_res.chunks)} inserted, got {inserted}"
    assert vdb.count_total_chunks() == inserted
    print(f"✅ Successfully inserted {inserted} chunks into ChromaDB.")

    # Test 6: Duplicate Upload Handling
    print("\n[Test 6] Testing Duplicate Document Detection...")
    assert vdb.is_document_indexed(result.document_id) is True, "Document should be indexed."
    # Attempting to re-add should skip
    re_inserted = vdb.add_chunks(chunk_res.chunks, embeddings)
    assert re_inserted == inserted
    assert vdb.count_total_chunks() == inserted, "Duplicate chunks were incorrectly added!"
    print("✅ Duplicate insertion avoided: vector database size remained constant.")

    # Test 7: Semantic Retrieval
    print("\n[Test 7] Testing Semantic Similarity Retrieval...")
    retriever = Retriever(vector_db=vdb, embedding_manager=embed_mgr)
    
    query = "How does Grover algorithm provide a speedup for searching?"
    ret_res = retriever.retrieve(query=query, top_k=3)
    assert ret_res.has_results is True, "Expected retrieval results."
    assert len(ret_res.chunks) > 0, "No chunks returned."
    
    # The top retrieved chunk should be from Page 3 where Grover's algorithm is described
    top_chunk = ret_res.chunks[0]
    assert top_chunk.page_number == 3, f"Expected Grover to match Page 3, got Page {top_chunk.page_number}"
    assert "Grover" in top_chunk.text, "Retrieved chunk does not contain expected keyword."
    assert top_chunk.similarity_score > 0.3, f"Similarity score too low: {top_chunk.similarity_score}"
    print(f"✅ Correctly retrieved Page {top_chunk.page_number} with similarity score {top_chunk.similarity_score:.4f}.")

    # Test 8: Document Deletion
    print("\n[Test 8] Testing Document Deletion from Vector Database...")
    del_ok = vdb.delete_document(result.document_id)
    assert del_ok is True, "Delete operation returned False."
    assert vdb.count_total_chunks() == 0, f"Expected 0 chunks remaining, got {vdb.count_total_chunks()}"
    print("✅ Successfully deleted document and all associated vectors.")

    # Test 9: LLM Generator & Missing API Key Handling
    print("\n[Test 9] Testing LLM Generator & Missing Key Handling...")
    llm_dummy = GeminiRAGGenerator(api_key="your_gemini_api_key_here")
    dummy_resp = llm_dummy.generate_response(query="Test question", retrieved_chunks=[top_chunk])
    assert dummy_resp.success is False, "Expected failure on placeholder key."
    assert "missing or invalid" in dummy_resp.answer.lower()
    print("✅ Handled placeholder/missing API key gracefully without crashing.")

    # Clean up test artifacts
    vdb.clear_collection()
    import shutil
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir, ignore_errors=True)
    print("✅ Cleaned up temporary test artifacts.")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
