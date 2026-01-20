"""
Hebrew Health Insurance RAG System
Main orchestration file for the RAG pipeline.
"""

import sys
from typing import List, Optional
from langchain_core.documents import Document

from src.parser import DocumentProcessor
from src.vectordb import VectorDB


class InsuranceRAG:
    """
    Main RAG system orchestrator for Hebrew health insurance documents.
    """

    def __init__(
        self,
        pdf_directory: str = "data/pdfs",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        persist_directory: str = "data/chroma_db"
    ):
        """
        Initialize the RAG system.

        Args:
            pdf_directory: Directory containing PDF files
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist the vector database
        """
        self.pdf_directory = pdf_directory
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectordb = VectorDB(persist_directory=persist_directory)
        self.documents: List[Document] = []

    def load_documents(self) -> List[Document]:
        """
        Load and process all PDF documents from the configured directory.

        Returns:
            List of processed LangChain Document objects
        """
        print(f"Loading documents from {self.pdf_directory}...")
        self.documents = self.processor.process_directory(self.pdf_directory)
        
        if self.documents:
            print(f"✓ Loaded {len(self.documents)} chunks")
        
        return self.documents

    def embed_and_store(self) -> int:
        """
        Embed documents and store in the vector database.

        Returns:
            Number of documents stored
        """
        if not self.documents:
            print("No documents to embed. Run load_documents() first.")
            return 0

        print(f"Embedding {len(self.documents)} documents...")
        count = self.vectordb.add_documents(self.documents)
        print(f"✓ Stored {count} documents in vector database")
        return count

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query in natural language
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        return self.vectordb.search(query, k=k, filter=filter)

    def get_stats(self) -> dict:
        """
        Get statistics about the loaded documents.

        Returns:
            Dictionary with document statistics
        """
        if not self.documents:
            return {"total_chunks": 0, "text_chunks": 0, "table_chunks": 0, "sources": 0}

        text_chunks = sum(1 for doc in self.documents if doc.metadata.get("content_type") == "text")
        table_chunks = sum(1 for doc in self.documents if doc.metadata.get("content_type") == "table")
        
        return {
            "total_chunks": len(self.documents),
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "sources": len(set(doc.metadata.get("source", "") for doc in self.documents))
        }


def main():
    """
    Main execution function - demonstrates the RAG pipeline.
    
    Usage:
        python rag.py           # Use existing DB if available
        python rag.py --reset   # Clear and rebuild database
    """
    rag = InsuranceRAG()
    
    # Check for reset flag
    force_reset = "--reset" in sys.argv
    db_exists = rag.vectordb.exists()
    
    if force_reset or not db_exists:
        if force_reset:
            print("🔄 Resetting database...")
            rag.vectordb.clear()
        else:
            print("📦 Building database from scratch...")
        
        # Load and process documents
        rag.load_documents()
        
        # Embed and store
        rag.embed_and_store()
        
        # Show statistics
        stats = rag.get_stats()
        print(f"\nStatistics:")
        print(f"  Sources: {stats['sources']}")
        print(f"  Text chunks: {stats['text_chunks']}")
        print(f"  Table chunks: {stats['table_chunks']}")
        print(f"  Total: {stats['total_chunks']}")
    else:
        count = rag.vectordb.count()
        print(f"✓ Database already exists with {count} documents")
        print(f"  Use 'python rag.py --reset' to rebuild")


if __name__ == "__main__":
    main()
