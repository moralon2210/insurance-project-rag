# Hebrew Health Insurance RAG System

A Retrieval-Augmented Generation solution that answers natural-language questions about your personal Hebrew health insurance papers. Upload your PDF policies, and the system ground responses in the source documents—coverage, reimbursement, exclusions, and pre-approval rules are all searchable in plain Hebrew.

## Pipeline Overview

![RAG Pipeline](RAG_pipeline.png)

### System Flow

```
Add PDFs → Process Documents → Chat Q&A grounded in your insurance policy
```

### Processing Stages

```
Upload PDF → Parse & Extract → Chunk Text → Generate Embeddings → Store in Vector DB → Hybrid Search (Vector + BM25) → Hebrew Reranking → LLM Answer
```

## Architecture & Components

| Component | Technology | Role |
|-----------|------------|------|
| **PDF Parsing** | `pdfplumber` | Hebrew-aware extraction with support for tables and multi-column layouts. |
| **Text Chunking** | LangChain text splitters | Recursive splitting tuned for Hebrew punctuation and grammar. |
| **Embeddings** | HuggingFace `sentence-transformers` | Multilingual E5 model running locally on CPU for semantic indexing. |
| **Vector DB** | ChromaDB | Persistent storage for vector embeddings. |
| **Retrieval** | Hybrid Search (Vector + BM25) | Combines semantic search with keyword matching for comprehensive recall. |
| **Reranking** | DictaBERT CrossEncoder | Hebrew-optimized reranker for precision refinement of search results. |
| **LLM Agent** | OpenAI GPT-4o-mini | Context-aware Hebrew prompts that remain faithful to the source documents. |
| **Orchestration** | LangChain scripts | Glue for parsing, embeddings, retrieval, and Q&A. |

## Project Layout

```
insurance-project-rag/
├── data/
│   ├── pdfs/                  # Drop your Hebrew insurance PDFs here
├── src/
│   ├── utils/
│   │   ├── __init__.py         # Shared exports
│   │   ├── embeddings.py       # Embedding helpers + caching logic
│   │   ├── pdf_parser.py       # Unified Hebrew PDF extraction (tables, images, fonts)
│   │   ├── processor.py        # Document processing orchestration (split, embed, persist)
│   │   ├── text_splitter.py    # Hebrew-optimized chunking utilities
│   │   └── tokenizer.py        # Token-count helpers for chunk compliance with LLM limits
│   ├── vectordb.py             # Vector DB + retrieval API surface
│   └── llm.py                  # LLM prompts, chat logic, and response grounding
├── tests/
│   ├── __init__.py
│   ├── test_parser.py          # Integrated parser + chunker tests
│   └── test_parser_simple.py   # Lightweight parser smoke tests
├── rag.py                      # CLI entry point for ingestion and chat
├── requirements.txt
└── README.md
```

## Features

- Hebrew-focused PDF parsing with tables and multi-column fallbacks.
- Recursive chunk splitting that respects Hebrew semantics while keeping token counts manageable.
- **Hybrid retrieval** combining semantic vector search (E5 embeddings) with keyword-based BM25 for comprehensive document recall.
- **Hebrew reranking** using DictaBERT CrossEncoder to refine and re-score retrieved candidates for optimal relevance.
- GPT-4o-mini Q&A prompts that cite the original policy language for compliance.
- Simple CLI (`python rag.py`) for database creation, query, and optional rebuilds.

## Getting Started

### 1. Clone & enter the repo

```bash
git clone <repository-url>
cd insurance-project-rag
```

### 2. Set up Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file at the project root:

```
OPENAI_API_KEY=your-api-key-here
```

### 5. Prepare documents

1. Place your Hebrew policy PDFs under `data/pdfs/`.
2. (Optional) Use the curated samples in `data/pdf_2/` for manual tests.

### 6. Run the RAG pipeline

```bash
# Build vector db on first run
python rag.py

# Reuse existing db
python rag.py

# Rebuild the db if documents change
python rag.py --reset
```

The CLI processes the PDFs, stores embeddings in Chroma, and launches a chat interface for querying.

## Testing

- Run unit tests with `pytest tests/`.
- Use `tests/test_parser_simple.py` for quick parser sanity checks before large uploads.

## What to Ask

- Coverage questions like “האם השתלות שיניים כלולות בכיסוי?”
- Reimbursement limits such as “מה הפיצוי עבור טיפולי פיזיותרפיה?”
- Policy exclusions or pre-approval requirements for treatments abroad.
- Comparisons between policy options by referencing clauses from the uploaded documents.
