# -*- coding: utf-8 -*-
"""
Hebrew Health Insurance RAG System
Main orchestration file for the RAG pipeline.
"""

import os
import sys
from typing import List
from langchain_core.documents import Document

from src.utils import DocumentProcessor
from src.vectordb import VectorDB
from src.llm import InsuranceLLM


class InsuranceRAG:
    """
    Main RAG system orchestrator for Hebrew health insurance documents.
    """

    def __init__(
        self,
        pdf_directory: str = "data/pdfs",
        max_tokens: int = 450,
        token_overlap: int = 50,
        persist_directory: str = "data/chroma_db"
    ):
        """
        Initialize the RAG system.

        Args:
            pdf_directory: Directory containing PDF files
            max_tokens: Maximum tokens per chunk (default: 450)
            token_overlap: Token overlap between chunks (default: 50)
            persist_directory: Directory to persist the vector database
        """
        self.pdf_directory = pdf_directory
        self.processor = DocumentProcessor(
            max_tokens=max_tokens,
            token_overlap=token_overlap
        )
        self.vectordb = VectorDB(persist_directory=persist_directory)
        self.llm = None  # Lazy loading
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
            print(f"Loaded {len(self.documents)} chunks")
        
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

        print(f"\n[*] Starting database creation...")
        print(f"[*] Embedding {len(self.documents)} documents...")
        count = self.vectordb.add_documents(self.documents)
        print(f"[+] Database creation completed!")
        print(f"[+] Stored {count} documents in vector database")
        return count

    def query(self, question: str, k: int = 5, show_sources: bool = False) -> str:
        """
        Ask a question about the insurance documents.

        Args:
            question: Question in Hebrew
            k: Number of documents to retrieve for context
            show_sources: Whether to show sources in the answer

        Returns:
            Answer from the LLM
        """
        # Initialize LLM if not already done (lazy loading)
        if self.llm is None:
            self.llm = InsuranceLLM()
        
        # 1. Retrieve relevant documents
        print(f"\n[DEBUG] Searching for: {question}")
        documents = self.vectordb.search(question, k=k, use_reranker=True)
        print(f"[DEBUG] Found {len(documents)} documents (after reranking)")
        
        if documents:
            print("\n" + "="*80)
            print("[DEBUG] FULL RETRIEVED CHUNKS:")
            print("="*80)
            for i, doc in enumerate(documents, 1):
                source = os.path.basename(doc.metadata.get("source", "?"))
                page = doc.metadata.get("page", "?")
                content_type = doc.metadata.get("content_type", "?")
                print(f"\n--- CHUNK {i} ---")
                print(f"Source: {source}")
                print(f"Page: {page}")
                print(f"Type: {content_type}")
                print(f"Content length: {len(doc.page_content)} chars")
                print(f"Content:\n{doc.page_content}")
                print("-" * 80)
            print("="*80 + "\n")
        
        if not documents:
            return "No relevant documents found for this question."
        
        # 2. Format context
        context = self.llm.format_context(documents)
        
        # 3. Ask LLM
        answer = self.llm.ask(question, context)
        
        # 4. Add sources if requested
        if show_sources:
            answer += "\n\nSources:\n"
            for i, doc in enumerate(documents, 1):
                source = os.path.basename(doc.metadata.get("source", ""))
                page = doc.metadata.get("page", "?")
                answer += f"  {i}. {source} (page {page})\n"
        
        return answer

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
            print("Resetting database...")
            rag.vectordb.clear()
        else:
            print("Building database from scratch...")
        
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
        print(f"Database already exists with {count} documents")
        print(f"  Use 'python rag.py --reset' to rebuild")
    
    # Demonstrate the RAG pipeline with example questions
    print("\n" + "="*60)
    print("RAG Pipeline Demo - Chat Interface")
    print("Ask questions about the insurance documents")
    print("To quit, type 'q', 'exit', or 'quit', or press Ctrl+C")
    print("="*60 + "\n")
    
    while True:
        try:
            example_question = input("\nYour question: ").strip()
            # Check for exit commands
            if example_question.lower() in ['q', 'exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Skip empty input
            if not example_question:
                print("Please ask a question")
                continue
            
            print(f"\nQuestion: {example_question}")
            print("-" * 60)
            
            answer = rag.query(example_question, k=5, show_sources=True)
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
