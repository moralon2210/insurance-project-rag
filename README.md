# Hebrew Health Insurance RAG System

A RAG (Retrieval-Augmented Generation) system designed to **query and answer questions about Hebrew health insurance papers**.

This system processes complex Hebrew insurance documents (policies, coverage details, terms) and makes them searchable through a question-answering interface.

## Project Structure

```
project/
├── src/
│   └── parser/
│       ├── __init__.py          # Module exports
│       ├── pdf_parser.py        # Unified PDF extraction with table support
│       ├── text_splitter.py     # Hebrew-optimized text chunking
│       └── processor.py         # Document processing orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_parser.py           # Full test with interactive mode
│   └── test_parser_simple.py    # Quick test script
├── data/
│   └── pdfs/                    # Place your PDF files here
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Add a PDF file** to the `data/pdfs/` directory

2. **Run the test script**:
```bash
# Simple test - shows first 2 chunks
python -m tests.test_parser_simple

# Full test - with statistics and interactive mode
python -m tests.test_parser
```

### In Your Code

```python
from src.parser import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=600,
    chunk_overlap=120,
    table_aware=False  # Set True to keep table rows together
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
- **Metadata Preservation**: Keeps track of source file and page numbers

## How It Works

This system is built specifically for querying health insurance documents:

1. **PDF Parsing**: Extracts text from Hebrew insurance PDFs using character-level extraction to handle complex formatting
2. **Table Conversion**: Identifies and converts tables (coverage amounts, terms, conditions) to Markdown
3. **Text Chunking**: Splits content into semantic chunks optimized for Hebrew text
4. **Ready for RAG**: Produces LangChain documents ready for embedding and vector storage
5. **Query Interface**: (Coming soon) Natural language questions about coverage, terms, and conditions

### Use Cases

- "What is covered under policy 552?"
- "What are the deductibles for specialized medications?"
- "Is treatment X covered?"
- "What are the exclusions in this policy?"


