# 📚 Modular PDF-Based RAG Application

An end-to-end, modular, and beginner-friendly **Retrieval-Augmented Generation (RAG)** application built in Python. This project allows users to upload PDF documents, extract text page-by-page, split text into semantic chunks, generate local dense embeddings using Sentence Transformers, store and index them in a persistent ChromaDB vector database, and perform context-grounded conversational Q&A with Google Gemini.

---

## 📑 Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Project Architecture](#2-project-architecture)
3. [What is PDF Text Extraction?](#3-what-is-pdf-text-extraction)
4. [What is Chunking?](#4-what-is-chunking)
5. [Why Do We Use Chunk Overlap?](#5-why-do-we-use-chunk-overlap)
6. [What are Embeddings?](#6-what-are-embeddings)
7. [How Does Sentence Transformers Work?](#7-how-does-sentence-transformers-work)
8. [What is a Vector Database?](#8-what-is-a-vector-database)
9. [How Does ChromaDB Store and Retrieve Vectors?](#9-how-does-chromadb-store-and-retrieve-vectors)
10. [What is Semantic Similarity Search?](#10-what-is-semantic-similarity-search)
11. [How Does Gemini Generate Answers from Retrieved Context?](#11-how-does-gemini-generate-answers-from-retrieved-context)
12. [Complete End-to-End Data Flow](#12-complete-end-to-end-data-flow)
13. [Installation Instructions](#13-installation-instructions)
14. [Environment Setup](#14-environment-setup)
15. [How to Run the Application](#15-how-to-run-the-application)
16. [Troubleshooting & Common Questions](#16-troubleshooting--common-questions)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that augments Large Language Models (LLMs) with external, dynamic knowledge bases.

Standard LLMs suffer from two core limitations:
1. **Knowledge Cutoffs**: They cannot know information published after their training date.
2. **Hallucinations**: When asked about proprietary, private, or specialized documents, they tend to guess or invent convincing falsehoods.

RAG solves this by decoupling knowledge storage from language reasoning:
- **Retrieval**: When a user asks a question, the system searches an external vector database for the most relevant document passages.
- **Augmentation**: The retrieved passages are injected into the prompt as factual context.
- **Generation**: The LLM synthesizes an accurate answer grounded strictly in that retrieved context, citing actual sources.

---

## 2. Project Architecture

The application is structured into decoupled, modular components following software engineering best practices:

```text
PDF_RAG/
│
├── data/
│   └── uploads/              # Local cache of uploaded PDF documents
│
├── vectorstore/              # ChromaDB persistent on-disk database files
│
├── src/
│   ├── __init__.py           # Package initializer
│   ├── pdf_loader.py         # PyMuPDF text extraction & scanned PDF detection
│   ├── chunking.py           # RecursiveCharacterTextSplitter with page metadata
│   ├── embeddings.py         # Local Sentence Transformers embedding manager
│   ├── vector_db.py          # ChromaDB collection management, deduplication & CRUD
│   ├── retrieval.py          # Semantic cosine similarity search & thresholding
│   └── llm.py                # Google Gemini synthesis & citation generator
│
├── app.py                    # Streamlit interactive UI dashboard
├── config.py                 # Centralized configuration dataclass & env loader
├── requirements.txt          # Python package dependencies
├── .env.example              # Template for API keys
├── .gitignore                # Ignore vector stores, bytecode, environments, and secrets
└── README.md                 # Complete documentation and concept explanations
```

---

## 3. What is PDF Text Extraction?

PDF (Portable Document Format) is designed for fixed visual display and printing rather than semantic parsing. Behind the scenes, a PDF file contains rendering instructions: glyph positions, fonts, vectors, and bounding boxes.

### How `pdf_loader.py` Works:
1. **PyMuPDF (`fitz`)**: Directly inspects the PDF document structure and extracts characters associated with layout blocks.
2. **Page-by-Page Extraction**: Each page is parsed individually, preserving the exact page number (1-indexed) as metadata.
3. **Encrypted/Password Detection**: Detects `doc.is_encrypted` and alerts the user rather than failing silently.
4. **Scanned PDF Detection**: If a document contains pages with zero or negligible character counts despite containing images, the loader flags the document as scanned and clarifies that Optical Character Recognition (OCR) is required.

---

## 4. What is Chunking?

Large documents cannot be converted into a single embedding vector because:
- Embedding models have maximum token limits (e.g., 256 or 512 tokens).
- Averaging an entire 50-page document into a single vector washes out specific details.

**Chunking** breaks long texts into smaller, semantically coherent segments suitable for vector search.

In `src/chunking.py`, we use **`RecursiveCharacterTextSplitter`**:
- **Default Chunk Size**: 1,000 characters.
- **Default Chunk Overlap**: 200 characters.
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`. The splitter recursively tries to split at natural paragraph breaks first, then sentence breaks, then word spaces, only splitting across words as a last resort.
- **Page-Level Source Preservation**: Each page is chunked independently, ensuring that every chunk retains its parent document ID, file name, and exact page number.

---

## 5. Why Do We Use Chunk Overlap?

When text is split strictly at a boundary, sentences and crucial context can be cut in half:

```text
Without Overlap:
[Chunk 1: ... The total net profit of Acme Corp in 2024 was]
[Chunk 2: $45.2 million, representing an increase of 12% ...]
```
If a user asks: *"What was Acme Corp's net profit in 2024?"*, neither chunk contains the complete thought.

```text
With Overlap (e.g., 200 characters):
[Chunk 1: ... The total net profit of Acme Corp in 2024 was $45.2 million ...]
[Chunk 2: ... net profit of Acme Corp in 2024 was $45.2 million, representing an increase of 12% ...]
```
Overlap ensures continuity of thought, preserving semantic meaning across boundaries.

---

## 6. What are Embeddings?

An **embedding** is a translation of text into a high-dimensional vector (a list of numbers, e.g., 384 floating-point numbers) in a semantic space.

In this geometric space:
- Texts with similar meanings are positioned close together.
- For example:
  - Vector("The dog chased the ball") $\approx$ Vector("A puppy ran after the sphere")
  - Vector("Financial balance sheet") is far away from Vector("Chocolate cake recipe")

---

## 7. How Does Sentence Transformers Work?

Sentence Transformers uses a Siamese neural network architecture based on BERT models trained with contrastive learning.

1. **Local Computation**: In `src/embeddings.py`, the model `all-MiniLM-L6-v2` is downloaded once and cached locally. Embeddings run on your CPU/GPU without sending your data to any paid external API.
2. **Fixed-Size Representation**: Regardless of chunk length (up to the token limit), the model outputs a 384-dimensional vector.
3. **L2 Normalization**: The vectors are normalized so their Euclidean length equals 1. This allows cosine similarity to be computed via dot product.

---

## 8. What is a Vector Database?

Traditional relational databases (PostgreSQL, MySQL) index by exact values, keywords, or ranges using B-Trees. They cannot search for *meaning*.

A **Vector Database** is optimized to store millions of high-dimensional vectors and retrieve nearest neighbors in milliseconds using Approximate Nearest Neighbor (ANN) index structures (such as HNSW — Hierarchical Navigable Small World graphs).

---

## 9. How Does ChromaDB Store and Retrieve Vectors?

`src/vector_db.py` uses **ChromaDB**:
- **Persistent Storage**: Data is saved to `./vectorstore/` via `chromadb.PersistentClient()`.
- **Deduplication**: Before inserting chunks, the system calculates a content hash (`document_id`). If the document already exists, re-insertion is skipped.
- **Schema Stored Per Chunk**:
  - `id`: Deterministic unique ID (`{doc_id}_p{page}_c{chunk_index}`)
  - `document`: Raw text content
  - `embedding`: 384-dimensional float vector
  - `metadata`: `{"document_id": "...", "document_name": "...", "page_number": 3, "chunk_index": 5}`

---

## 10. What is Semantic Similarity Search?

When a user submits a query:
1. `Retriever` encodes the query into vector $V_q$ using the same Sentence Transformer model.
2. ChromaDB calculates the **Cosine Similarity** between $V_q$ and all stored chunk vectors $V_c$:
   $$\text{Cosine Similarity}(V_q, V_c) = \frac{V_q \cdot V_c}{\|V_q\| \|V_c\|}$$
3. The top-$K$ chunks with highest similarity (or lowest distance) are returned.
4. An optional similarity threshold (e.g., 0.3) filters out irrelevant matches.

---

## 11. How Does Gemini Generate Answers from Retrieved Context?

In `src/llm.py`, the retrieved chunks are formatted into a prompt for Google Gemini (`gemini-1.5-flash` or `gemini-2.0-flash`):

```text
[System Instructions: You are a strict factual document assistant...]

Context from uploaded documents:
--- CONTEXT BLOCK 1 ---
Document: quarterly_report.pdf
Page: 4
Content: Revenue grew by 18% year-over-year to $120 million.

User Question: What was the revenue growth?
```

Gemini reads the supplied context, extracts the exact facts, cites `quarterly_report.pdf (Page 4)`, and refuses to guess if the information is absent.

---

## 12. Complete End-to-End Data Flow

```text
       ┌──────────────┐
       │ Uploaded PDF │
       └──────┬───────┘
              │
              ▼
   [ PyMuPDF Text Extraction ] ────► Detect empty/scanned pages & preserve page #
              │
              ▼
  [ RecursiveCharacterSplitter ] ──► 1000 char chunks, 200 overlap + metadata
              │
              ▼
  [ SentenceTransformer Model ] ──► Local 384-d dense embedding vectors
              │
              ▼
      [ ChromaDB Storage ] ────────► Persistent vector store (./vectorstore)
              ▲
              │ Semantic Search (Cosine)
              ▼
       ┌──────────────┐
       │  User Query  │ ───────────► Embed query with all-MiniLM-L6-v2
       └──────────────┘
              │
              ▼
      [ Retrieved Chunks ] ────────► Top-K context blocks + page citations
              │
              ▼
     [ Google Gemini LLM ] ────────► Context-grounded answer + citations
```

---

## 13. Installation Instructions

### Prerequisites
- Python 3.11 or Python 3.12
- Git

### 1. Clone or Open the Repository
```bash
cd "lecture 2"
```

### 2. Create and Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 14. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here
```
> **How to get a key:** Visit [Google AI Studio](https://aistudio.google.com/) and click **Get API key**. It is free to use.

Alternatively, you can enter your API key directly in the Streamlit sidebar at runtime.

---

## 15. How to Run the Application

Start the Streamlit web dashboard:
```bash
streamlit run app.py
```

The browser will open automatically at:
```text
http://localhost:8501
```

### Quick Walkthrough:
1. Go to the **📤 Upload & Process PDFs** tab.
2. Drag and drop any text-based PDF file.
3. Click **🚀 Process & Index Documents**.
4. Switch to the **💬 Chat & Ask Questions** tab and ask questions about your documents!
5. Expand the **📑 Retrieved Sources** dropdown beneath each answer to view source pages and exact excerpts.
6. Use the **🔍 Vector Database Explorer** tab to inspect all chunks indexed in ChromaDB.

---

## 16. Troubleshooting & Common Questions

### Q: Why do I see "Warning: appears to be a scanned document"?
**A:** Scanned PDFs are images inside a PDF wrapper without digital text glyphs. PyMuPDF extracts embedded text. For scanned images, an OCR engine (such as Tesseract or Google Document AI) is needed.

### Q: Does the app re-embed documents when I restart the server?
**A:** No. ChromaDB saves all vectors to the `./vectorstore/` directory. Uploaded documents remain indexed between restarts, and duplicate uploads are automatically detected and skipped.

### Q: How do I delete indexed documents?
**A:** In the left sidebar under **📁 Stored Documents**, click the **🗑️ Delete** button next to any document, or click **🧹 Clear Entire Vector DB**.

### Q: Can I run this completely offline?
**A:** Text extraction (`PyMuPDF`), chunking (`LangChain`), and embedding generation (`SentenceTransformers`) all run 100% locally on your machine without internet access. Only the final Gemini API call requires internet connectivity.
