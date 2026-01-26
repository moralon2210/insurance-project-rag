"""
Simple test script for quick PDF parsing debugging
Run: python -m tests.test_parser_simple
"""

import os
import sys

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import DocumentProcessor


def main():
    # Find first PDF
    pdf_dir = r'data\pdfs'
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in data/pdfs/")
        print("Please add a PDF file to the data/pdfs/ directory")
        return
    
    pdf_path = os.path.join(pdf_dir, pdf_files[0])
    print(f"📄 Processing: {pdf_path}\n")
    
    # Process
    processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
    documents = processor.process_file(pdf_path)
    
    print(f"✅ Generated {len(documents)} chunks")
    
    # Analyze all chunks using the new content_type metadata
    chunks_with_tables = []
    chunks_without_tables = []
    
    for i, doc in enumerate(documents):
        content_type = doc.metadata.get("content_type", "unknown")
        if content_type == "table":
            chunks_with_tables.append((i, doc))
        else:
            chunks_without_tables.append((i, doc))
    
    print(f"📊 Chunks with tables: {len(chunks_with_tables)}")
    print(f"📝 Chunks with text only: {len(chunks_without_tables)}\n")
    
    # Show first text-only chunk
    if chunks_without_tables:
        i, doc = chunks_without_tables[0]
        print(f"\n{'='*70}")
        print(f"📝 EXAMPLE TEXT CHUNK #{i} | Page {doc.metadata.get('page')} | {len(doc.page_content)} chars")
        print('='*70)
        print(doc.page_content[:500])
        if len(doc.page_content) > 500:
            print("\n... (truncated)")
    
    # Show first 2 table chunks
    num_to_show = min(2, len(chunks_with_tables))
    for idx in range(num_to_show):
        i, doc = chunks_with_tables[idx]
        print(f"\n{'='*70}")
        print(f"📊 EXAMPLE TABLE CHUNK #{i} | Page {doc.metadata.get('page')} | {len(doc.page_content)} chars")
        print('='*70)
        print(doc.page_content)
    
    # Show pages where tables appear
    if chunks_with_tables:
        table_pages = sorted(set(doc.metadata.get('page', 0) for _, doc in chunks_with_tables))
        print(f"\n\n💡 Tables found on pages: {table_pages}")
    
    print(f"💡 Total chunks: {len(documents)}")
    print("💡 For interactive exploration, use: python -m tests.test_parser")


if __name__ == "__main__":
    main()
