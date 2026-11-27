"""
Streamlit Web Application for PDF-Based RAG Application.
Provides an interactive dashboard for PDF uploading, chunking, embedding generation,
persistent vector management, and context-grounded conversational QA with Google Gemini.
"""

import os
from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from config import config, DATA_DIR, VECTOR_DB_DIR
from src.pdf_loader import PDFLoader, PDFExtractionResult
from src.chunking import DocumentChunker, ChunkingResult
from src.embeddings import EmbeddingManager
from src.vector_db import VectorDatabase
from src.retrieval import Retriever
from src.llm import GeminiRAGGenerator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN STYLING ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .citation-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #4338CA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CACHED SINGLETONS ---
@st.cache_resource(show_spinner=False)
def get_vector_db():
    return VectorDatabase(
        persist_directory=config.chroma_persist_dir,
        collection_name=config.chroma_collection_name,
    )

@st.cache_resource(show_spinner=False)
def get_embedding_manager(model_name: str):
    return EmbeddingManager.get_instance(model_name=model_name)

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ RAG Settings")

    # API Key Configuration
    st.markdown("### 🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key:",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Get a free key at https://aistudio.google.com/",
    )
    if api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input.strip()

    if st.session_state.gemini_api_key and not st.session_state.gemini_api_key.startswith("your_"):
        st.success("API Key detected ✅", icon="🔒")
    else:
        st.warning("Please configure your Gemini API Key.", icon="⚠️")

    st.markdown("---")

    # Model & Retrieval Settings
    st.markdown("### 🧠 Model Parameters")
    model_options = [
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ]
    selected_gemini_model = st.selectbox(
        "Gemini LLM Model:",
        options=model_options,
        index=0,
        help="Recommended: gemini-2.5-flash or gemini-3.5-flash / gemini-3.8-flash",
    )

    selected_embedding_model = st.selectbox(
        "Embedding Model (Local):",
        options=config.available_embedding_models,
        index=0,
        help="Runs 100% locally on your machine via Sentence Transformers.",
    )

    top_k_slider = st.slider(
        "Top-K Retrieved Chunks:",
        min_value=1,
        max_value=15,
        value=config.default_top_k,
        step=1,
        help="Number of most relevant text chunks sent as context to Gemini.",
    )

    sim_threshold = st.slider(
        "Min Similarity Threshold:",
        min_value=0.0,
        max_value=0.9,
        value=0.0,
        step=0.05,
        help="Filter chunks with cosine similarity below this score (0.0 = no filter).",
    )

    with st.expander("✂️ Chunking Parameters"):
        chunk_size_input = st.number_input(
            "Chunk Size (characters):",
            min_value=200,
            max_value=3000,
            value=config.chunk_size,
            step=100,
        )
        chunk_overlap_input = st.number_input(
            "Chunk Overlap (characters):",
            min_value=0,
            max_value=500,
            value=config.chunk_overlap,
            step=50,
        )
        if chunk_overlap_input >= chunk_size_input:
            st.error("Overlap must be strictly smaller than Chunk Size!")

    st.markdown("---")

    # Document Management Section in Sidebar
    st.markdown("### 📁 Stored Documents")
    vector_db = get_vector_db()
    stored_docs = vector_db.get_all_stored_documents()
    total_db_chunks = vector_db.count_total_chunks()

    col_side1, col_side2 = st.columns(2)
    with col_side1:
        st.metric("Documents", len(stored_docs))
    with col_side2:
        st.metric("Total Chunks", total_db_chunks)

    if stored_docs:
        for doc in stored_docs:
            with st.expander(f"📄 {doc.document_name[:20]}...", expanded=False):
                st.write(f"**ID:** `{doc.document_id[:12]}...`")
                st.write(f"**Pages:** {doc.page_count} (indexed: {doc.pages_indexed[:5]}...)")
                st.write(f"**Chunks:** {doc.chunk_count}")
                if st.button("🗑️ Delete", key=f"del_{doc.document_id}", use_container_width=True):
                    success = vector_db.delete_document(doc.document_id)
                    if success:
                        st.toast(f"Deleted '{doc.document_name}'", icon="🗑️")
                        st.rerun()

        if st.button("🧹 Clear Entire Vector DB", type="secondary", use_container_width=True):
            vector_db.clear_collection()
            st.session_state.messages = []
            st.toast("Vector database cleared!", icon="🧹")
            st.rerun()
    else:
        st.info("No documents stored in vector database.")

# --- MAIN PAGE LAYOUT ---
st.markdown('<div class="main-header">📚 PDF-Based RAG Application</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload PDFs, split into semantic chunks, embed locally with Sentence Transformers, '
    'and query with Google Gemini grounded strictly in context.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_upload, tab_explorer = st.tabs(
    ["💬 Chat & Ask Questions", "📤 Upload & Process PDFs", "🔍 Vector Database Explorer"]
)

# ==========================================
# TAB 2: UPLOAD & PROCESS DOCUMENTS
# ==========================================
with tab_upload:
    st.subheader("Upload PDF Documents")
    st.write(
        "Upload one or multiple PDF documents. PyMuPDF extracts text per page, preserves page metadata, "
        "and flags scanned documents."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files to index:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard PDF files with selectable text. Password-protected files will be rejected.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s) for processing.")

        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            pdf_loader = PDFLoader()
            chunker = DocumentChunker(
                chunk_size=chunk_size_input,
                chunk_overlap=chunk_overlap_input,
            )
            embedding_manager = get_embedding_manager(selected_embedding_model)

            progress_bar = st.progress(0.0)
            status_box = st.empty()

            total_files = len(uploaded_files)
            new_chunks_count = 0
            new_chars_count = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                status_box.markdown(f"⏳ **[{idx+1}/{total_files}] Reading `{file_name}`...**")

                # Extract text using PyMuPDF
                save_path = DATA_DIR / file_name
                extraction = pdf_loader.load_pdf(
                    file_source=uploaded_file.getvalue(),
                    filename=file_name,
                    save_copy_path=save_path,
                )

                if not extraction.success:
                    st.error(f"❌ Failed to extract '{file_name}': {extraction.error_message}")
                    continue

                if extraction.warning_message:
                    st.warning(f"⚠️ {file_name}: {extraction.warning_message}")

                # Check for duplicate document
                if vector_db.is_document_indexed(extraction.document_id):
                    existing_count = vector_db.get_document_chunk_count(extraction.document_id)
                    st.info(
                        f"ℹ️ **'{file_name}'** is already indexed ({existing_count} chunks). Skipping duplicate storage."
                    )
                    progress_bar.progress((idx + 1) / total_files)
                    continue

                # Chunk document
                status_box.markdown(f"✂️ **[{idx+1}/{total_files}] Chunking `{file_name}`...**")
                chunk_result: ChunkingResult = chunker.chunk_extraction_result(extraction)

                if not chunk_result.chunks:
                    st.warning(
                        f"⚠️ Document '{file_name}' had {extraction.total_pages} pages but no indexable text was found."
                    )
                    continue

                # Generate embeddings locally
                status_box.markdown(
                    f"🧠 **[{idx+1}/{total_files}] Generating embeddings for {chunk_result.total_chunks} chunks...**"
                )
                chunk_texts = [c.text for c in chunk_result.chunks]
                embeddings = embedding_manager.embed_texts(chunk_texts, batch_size=32)

                # Store in ChromaDB
                status_box.markdown(
                    f"💾 **[{idx+1}/{total_files}] Storing chunks in ChromaDB...**"
                )
                inserted = vector_db.add_chunks(chunk_result.chunks, embeddings)

                new_chunks_count += inserted
                new_chars_count += chunk_result.total_characters

                st.success(
                    f"✅ **{file_name}**: Successfully indexed {extraction.total_pages} pages, "
                    f"{chunk_result.total_characters:,} characters into **{inserted} chunks**!"
                )

                progress_bar.progress((idx + 1) / total_files)

            status_box.markdown("✨ **All documents processed successfully!**")
            time.sleep(1)
            st.rerun()

# ==========================================
# TAB 1: RAG CHAT INTERFACE
# ==========================================
with tab_chat:
    # Top stats bar
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(stored_docs)}</div>
                <div class="metric-label">Indexed Documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_db_chunks}</div>
                <div class="metric-label">Total Stored Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{top_k_slider}</div>
                <div class="metric-label">Top-K Context Depth</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not stored_docs:
        st.info(
            "👋 Welcome! To begin, switch to the **📤 Upload & Process PDFs** tab to upload and index documents.",
            icon="ℹ️",
        )

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📑 Referenced Sources & Citations", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"- **{cit['document_name']}** (Page {cit['page_number']}) — "
                            f"<span class='badge'>Similarity: {cit['similarity_score']:.2f}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f'"{cit["excerpt"]}"')

    # Query Input
    user_query = st.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        # Display user query in chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if total_db_chunks == 0:
                warning_text = (
                    "No documents have been indexed yet. Please upload and process a PDF "
                    "in the 'Upload & Process PDFs' tab first."
                )
                st.warning(warning_text)
                st.session_state.messages.append({"role": "assistant", "content": warning_text})
            else:
                embedding_manager = get_embedding_manager(selected_embedding_model)
                retriever = Retriever(vector_db=vector_db, embedding_manager=embedding_manager)
                llm_generator = GeminiRAGGenerator(
                    api_key=st.session_state.gemini_api_key,
                    model_name=selected_gemini_model,
                )

                with st.spinner("🔍 Performing semantic search across ChromaDB chunks..."):
                    retrieval_res = retriever.retrieve(
                        query=user_query,
                        top_k=top_k_slider,
                        similarity_threshold=sim_threshold,
                    )

                if not retrieval_res.has_results:
                    no_info_msg = (
                        "I could not find any relevant information in the uploaded documents to answer your question. "
                        "(Similarity scores were below the set threshold or no matching text was found)."
                    )
                    st.info(no_info_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_info_msg})
                else:
                    with st.spinner(f"✨ Generating grounded response with {selected_gemini_model}..."):
                        llm_res = llm_generator.generate_response(
                            query=user_query,
                            retrieved_chunks=retrieval_res.chunks,
                        )

                    st.markdown(llm_res.answer)

                    # Display retrieved context and citations
                    if llm_res.citations:
                        with st.expander(f"📑 Retrieved Sources ({len(llm_res.citations)} citations)", expanded=True):
                            for cit in llm_res.citations:
                                st.markdown(
                                    f"**📄 {cit.document_name}** | **Page {cit.page_number}** | "
                                    f"<span class='badge'>Similarity: {cit.similarity_score:.2f}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f'<div class="citation-box">{cit.excerpt}</div>', unsafe_allow_html=True)

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": llm_res.answer,
                            "citations": [
                                {
                                    "document_name": c.document_name,
                                    "page_number": c.page_number,
                                    "similarity_score": c.similarity_score,
                                    "excerpt": c.excerpt,
                                }
                                for c in llm_res.citations
                            ],
                        }
                    )

# ==========================================
# TAB 3: VECTOR DATABASE EXPLORER
# ==========================================
with tab_explorer:
    st.subheader("ChromaDB Vector Store Explorer")
    st.write(
        "Inspect the raw chunks, metadata, and embeddings stored in your persistent ChromaDB vector database."
    )

    if not stored_docs:
        st.info("No vectors stored in database yet.")
    else:
        doc_names = {d.document_id: f"{d.document_name} ({d.chunk_count} chunks)" for d in stored_docs}
        selected_doc_id = st.selectbox(
            "Select a document to inspect its chunks:",
            options=list(doc_names.keys()),
            format_func=lambda x: doc_names[x],
        )

        if selected_doc_id:
            chunks_data = vector_db.collection.get(
                where={"document_id": selected_doc_id},
                include=["documents", "metadatas"],
            )

            ids = chunks_data.get("ids", [])
            docs = chunks_data.get("documents", [])
            metas = chunks_data.get("metadatas", [])

            st.write(f"Total chunks retrieved for this document: **{len(ids)}**")

            for i, (cid, text, meta) in enumerate(zip(ids, docs, metas)):
                page = meta.get("page_number", 1) if meta else 1
                c_idx = meta.get("chunk_index", i) if meta else i
                char_c = len(text)

                with st.expander(f"Chunk #{c_idx + 1} | Page {page} | ID: `{cid}` ({char_c} chars)"):
                    st.text(text)
                    st.json(meta)
