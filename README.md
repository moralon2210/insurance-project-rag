# Hebrew Health Insurance RAG System

A RAG (Retrieval-Augmented Generation) system that lets you **ask questions about your private health insurance policy in natural language**.

Get instant answers about your coverage, reimbursements, and terms - simply upload your Hebrew insurance documents and ask questions like "Is this treatment covered?" or "How much reimbursement will I get?"

## RAG Pipeline

![RAG Pipeline](RAG_pipeline.png)

## RAG Logic

**User Flow**
```
Add PDFs → Process Documents → Chat Q&A grounded in your insurance policy
```

**Processing Flow**
```
Upload PDF → Parse & Extract → Chunk Text → Generate Embeddings → Store in Vector DB → Semantic Search → LLM Answer
```

## Architecture

| Component | Technology | Description |
|-----------|------------|-------------|
| **PDF Parsing** | pdfplumber | Character-level Hebrew extraction with table support |
| **Chunking** | LangChain Text Splitters | Hebrew-optimized recursive splitting |
| **Embeddings** | HuggingFace sentence-transformers | Multilingual model running locally on CPU |
| **Vector DB** | ChromaDB | Persistent local storage with semantic search |
| **LLM** | OpenAI GPT-4o-mini | Hebrew-optimized prompts for accurate answers |
| **Framework** | LangChain | Document processing and chain orchestration |

## Project Structure

```
project/
├── src/
│   ├── utils/
│   │   ├── __init__.py          # Module exports
│   │   ├── pdf_parser.py        # Unified PDF extraction with table support
│   │   ├── text_splitter.py     # Hebrew-optimized text chunking
│   │   └── processor.py         # Document processing orchestrator
│   ├── vectordb.py              # Vector DB + Retrieval
│   └── llm.py                   # LLM integration & prompts
├── tests/
│   ├── __init__.py
│   ├── test_parser.py           # Full test with interactive mode
│   └── test_parser_simple.py    # Quick test script
├── data/
│   ├── pdfs/                    # Place your PDF files here
│   └── chroma_db/               # Vector database storage (auto-created)
├── rag.py                       # Main RAG orchestration file
├── .env                         # API keys (create from .env.example)
├── requirements.txt
└── README.md
```

## Getting Started

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd insurance-project-rag
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### Step 5: Add PDFs and Run

1. **Add PDF files** to the `data/pdfs/` directory

2. **Run the RAG system**:
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

## Usage

### Using the RAG System

```python
from rag import InsuranceRAG

# Initialize the RAG system
rag = InsuranceRAG()

# Build the database (first run)
rag.load_documents()
rag.embed_and_store()

# Ask questions in natural language
answer = rag.query("האם יש כיסוי להשתלות בחו\"ל?")
print(answer)

# With sources
answer = rag.query("כמה מקבלים החזר על פיזיותרפיה?", show_sources=True)
print(answer)

# Search for similar documents (raw search via vectordb)
results = rag.vectordb.search("כיסוי רפואי", k=5)
for doc in results:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
```

### Using the Processor Directly

```python
from src.utils import DocumentProcessor

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
- **Intelligent Retrieval**: Context-aware document retrieval and formatting
- **Hebrew LLM Integration**: OpenAI GPT with Hebrew-optimized system prompt
- **Metadata Preservation**: Keeps track of source file and page numbers

## How It Works

This system helps you understand your private health insurance policy:

1. **PDF Parsing**: Extracts text from Hebrew insurance PDFs using character-level extraction to handle complex formatting
2. **Table Conversion**: Identifies and converts tables (coverage amounts, terms, conditions) to Markdown
3. **Text Chunking**: Splits content into semantic chunks optimized for Hebrew text
4. **Embedding**: Converts chunks to vectors using a multilingual model (supports Hebrew)
5. **Vector Storage**: Stores embeddings in ChromaDB for fast semantic search
6. **Retrieval**: Finds relevant document chunks based on semantic similarity
7. **Answer Generation**: Uses OpenAI GPT with context to answer questions accurately in Hebrew

### What You Can Ask

- **Coverage Questions**: "Is dental implant treatment covered in my policy?"
- **Reimbursement Details**: "How much reimbursement do I get for physiotherapy sessions?"
- **Exclusions**: "Are cosmetic procedures excluded from coverage?"
- **Specialist Care**: "Do I need pre-approval for this specialist treatment abroad?"

The system analyzes your actual insurance documents to provide personalized answers based on your specific policy terms and coverage.


