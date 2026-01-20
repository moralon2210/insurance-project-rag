# Hebrew Health Insurance RAG System

A RAG (Retrieval-Augmented Generation) system that lets you **ask questions about your private health insurance policy in natural language**.

Get instant answers about your coverage, reimbursements, and terms - simply upload your Hebrew insurance documents and ask questions like "Is this treatment covered?" or "How much reimbursement will I get?"

## Project Structure

```
project/
├── src/
│   ├── parser/
│   │   ├── __init__.py          # Module exports
│   │   ├── pdf_parser.py        # Unified PDF extraction with table support
│   │   ├── text_splitter.py     # Hebrew-optimized text chunking
│   │   └── processor.py         # Document processing orchestrator
│   └── vectordb.py              # Vector database with local embeddings
├── tests/
│   ├── __init__.py
│   ├── test_parser.py           # Full test with interactive mode
│   └── test_parser_simple.py    # Quick test script
├── data/
│   ├── pdfs/                    # Place your PDF files here
│   └── chroma_db/               # Vector database storage (auto-created)
├── rag.py                       # Main RAG orchestration file
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Add PDF files** to the `data/pdfs/` directory

2. **Run the main RAG system**:
```bash
# First run - builds the database
python rag.py

# Subsequent runs - uses existing database
python rag.py

# Force rebuild - clears and rebuilds database
python rag.py --reset
```

Or run the test scripts:
```bash
# Simple test - shows first 2 chunks
python -m tests.test_parser_simple

# Full test - with statistics and interactive mode
python -m tests.test_parser
```

### Using the RAG System

```python
from rag import InsuranceRAG

# Initialize the RAG system
rag = InsuranceRAG(pdf_directory="data/pdfs")

# Clear and rebuild database (optional)
rag.vectordb.clear()

# Load and embed documents
rag.load_documents()
rag.embed_and_store()

# Search for similar documents
results = rag.search("האם יש כיסוי להשתלות בחו\"ל?", k=5)
for doc in results:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")

# Get statistics
stats = rag.get_stats()
print(f"Total: {stats['total_chunks']} chunks from {stats['sources']} sources")
```

### Using the Processor Directly

```python
from src.parser import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=600,
    chunk_overlap=120
)

# Process all PDFs in a directory
documents = processor.process_directory("data/pdfs")

# Process a single PDF
documents = processor.process_file("data/pdfs/insurance_policy.pdf")

# Documents are ready for embedding/vector store
for doc in documents:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

## Features

- **Character-Level Hebrew Extraction**: Custom parser that avoids text duplication issues common in Hebrew PDFs
- **Table Extraction**: Converts PDF tables to Markdown format for better LLM comprehension  
- **Deduplication**: Smart handling of PDFs with bold-effect character duplication
- **Optimized Chunking**: Hebrew-appropriate chunk sizes and separators
- **Local Embeddings**: Multilingual sentence-transformers model running entirely on CPU
- **Vector Database**: ChromaDB for persistent local storage with semantic search
- **Metadata Preservation**: Keeps track of source file and page numbers

## How It Works

This system helps you understand your private health insurance policy:

1. **PDF Parsing**: Extracts text from Hebrew insurance PDFs using character-level extraction to handle complex formatting
2. **Table Conversion**: Identifies and converts tables (coverage amounts, terms, conditions) to Markdown
3. **Text Chunking**: Splits content into semantic chunks optimized for Hebrew text
4. **Embedding**: Converts chunks to vectors using a multilingual model (supports Hebrew)
5. **Vector Storage**: Stores embeddings in ChromaDB for fast semantic search
6. **Semantic Search**: Query in natural language and get relevant document chunks

### What You Can Ask

- **Coverage Questions**: "Is dental implant treatment covered in my policy?"
- **Reimbursement Details**: "How much reimbursement do I get for physiotherapy sessions?"
- **Exclusions**: "Are cosmetic procedures excluded from coverage?"
- **Specialist Care**: "Do I need pre-approval for this specialist treatment abroad?"

The system analyzes your actual insurance documents to provide personalized answers based on your specific policy terms and coverage.


