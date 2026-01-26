"""
Test script for PDF parsing
Run from project root: python -m tests.test_parser
"""

import os
import sys
import json

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import DocumentProcessor


def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def test_parsing():
    """Test the PDF parsing pipeline"""
    
    # Find first PDF in data/pdfs folder
    pdf_dir = os.path.join("data", "pdfs")
    
    if not os.path.exists(pdf_dir):
        print(f"❌ Directory not found: {pdf_dir}")
        return
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Please add a PDF file to data/pdfs/ directory")
        return
    
    # Take first PDF
    pdf_path = os.path.join(pdf_dir, pdf_files[0])
    print_separator(f"Testing PDF: {os.path.basename(pdf_path)}")
    print(f"📄 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    # Initialize processor
    print("\n🔧 Initializing processor...")
    processor = DocumentProcessor(
        chunk_size=600,
        chunk_overlap=120
    )
    
    # Process the file
    print("🔍 Processing PDF...")
    try:
        documents = processor.process_file(pdf_path)
        print(f"✅ Success! Generated {len(documents)} chunks")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show statistics
    print_separator("STATISTICS")
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chunk_size = total_chars / len(documents) if documents else 0
    
    print(f"Total chunks: {len(documents)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chunk size: {avg_chunk_size:.0f} chars")
    
    # Show unique pages
    pages = set(doc.metadata.get('page', 0) for doc in documents)
    print(f"Pages processed: {sorted(pages)}")
    
    # Show first 3 chunks in detail
    print_separator("FIRST 3 CHUNKS (DETAILED)")
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- CHUNK #{i+1} ---")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Source: {os.path.basename(doc.metadata.get('source', ''))}")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"\nContent preview (first 300 chars):")
        print("-" * 40)
        print(doc.page_content[:300])
        print("-" * 40)
        
        # Check for tables
        if "|" in doc.page_content:
            print("📊 Contains table!")
            # Count table rows
            table_rows = doc.page_content.count("\n|")
            print(f"   Approximate table rows: {table_rows}")
    
    # Look for tables specifically
    print_separator("TABLE DETECTION")
    chunks_with_tables = [doc for doc in documents if "|" in doc.page_content]
    print(f"Chunks containing tables: {len(chunks_with_tables)}")
    
    if chunks_with_tables:
        print("\nFirst table found:")
        print("-" * 40)
        # Show first table chunk
        table_chunk = chunks_with_tables[0].page_content
        print(table_chunk[:500])
        print("-" * 40)
    
    # Show metadata structure
    print_separator("METADATA STRUCTURE")
    if documents:
        print("Sample metadata from first chunk:")
        print(json.dumps(documents[0].metadata, indent=2, ensure_ascii=False))
    
    # Interactive exploration option
    print_separator("INTERACTIVE MODE")
    print("\nEnter chunk number to view (0-{}), or 'q' to quit:".format(len(documents)-1))
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'q':
                break
            
            chunk_num = int(user_input)
            if 0 <= chunk_num < len(documents):
                doc = documents[chunk_num]
                print(f"\n--- CHUNK #{chunk_num} ---")
                print(f"Metadata: {doc.metadata}")
                print(f"\nFull content:")
                print("="*60)
                print(doc.page_content)
                print("="*60)
            else:
                print(f"❌ Invalid chunk number. Use 0-{len(documents)-1}")
        
        except ValueError:
            print("❌ Please enter a number or 'q'")
        except KeyboardInterrupt:
            break
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_parsing()
