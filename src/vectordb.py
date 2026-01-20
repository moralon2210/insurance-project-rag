"""
Vector Database for Hebrew Health Insurance RAG System

Combines local embeddings and ChromaDB storage in a single module.
Runs entirely locally on CPU.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorDB:
    """
    Local vector database with multilingual embeddings and ChromaDB storage.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_PERSIST_DIR = "data/chroma_db"
    DEFAULT_COLLECTION = "insurance_docs"

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        model_name: str = None
    ):
        """
        Initialize the vector database.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
            model_name: HuggingFace embedding model name
        """
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.model_name = model_name or self.DEFAULT_MODEL
        self._embeddings = None
        self._store = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the embedding model (creates it if needed).
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def get_store(self) -> Chroma:
        """
        Get the ChromaDB store (creates it if needed).
        """
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.get_embeddings(),
                persist_directory=self.persist_directory
            )
        return self._store

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Call Chroma's add_documents method
        store = self.get_store()
        store.add_documents(documents)
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        store = self.get_store()
        if filter:
            return store.similarity_search(query, k=k, filter=filter)
        return store.similarity_search(query, k=k)

    def count(self) -> int:
        """Get the number of documents in the store."""
        store = self.get_store()
        return store._collection.count()

    def clear(self):
        """Clear all documents from the store."""
        store = self.get_store()
        store._collection.delete(where={})
        self._store = None

    def exists(self) -> bool:
        """
        Check if the vector store directory exists with data.
        """
        if not os.path.exists(self.persist_directory):
            return False
        # Check if chroma.sqlite3 exists (the actual database file)
        db_file = os.path.join(self.persist_directory, "chroma.sqlite3")
        return os.path.exists(db_file)
