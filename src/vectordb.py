# -*- coding: utf-8 -*-
"""
Vector Database for Hebrew Health Insurance RAG System

Combines local embeddings and ChromaDB storage in a single module.
"""

import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever # Added import
from src.utils.embeddings import E5Embeddings

class VectorDB:
    DEFAULT_MODEL = "intfloat/multilingual-e5-base"
    DEFAULT_PERSIST_DIR = "data/chroma_db"
    DEFAULT_COLLECTION = "insurance_docs"

    def __init__(self, persist_directory: str = None, collection_name: str = None, model_name: str = None):
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.model_name = model_name or self.DEFAULT_MODEL
        
        self._embeddings = None
        self._store = None
        self._bm25 = None  # Lazy load BM25 retriever
        self._all_documents = None  # Store for BM25 indexing

    @property
    def embeddings(self):
        if self._embeddings is None:
            print("[*] Loading embedding model (first time only)...")
            self._embeddings = E5Embeddings(model_name=self.model_name, device="cpu", normalize_embeddings=True)
            print("[+] Embedding model loaded")
        return self._embeddings
    
    @property
    def store(self):
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._store

    def _get_bm25_retriever(self, initial_k: int):
        """Helper to initialize BM25 from memory or Chroma."""
        if self._bm25 is not None:
            self._bm25.k = initial_k
            return self._bm25

        # If we don't have documents in memory, pull them from Chroma
        if self._all_documents is None:
            print("[*] Extracting documents from Chroma for BM25...")
            data = self.store.get(include=['documents', 'metadatas'])
            self._all_documents = [
                Document(page_content=doc, metadata=meta) 
                for doc, meta in zip(data['documents'], data['metadatas'])
            ]

        if self._all_documents:
            print(f"[*] Initializing BM25 with {len(self._all_documents)} docs...")
            self._bm25 = BM25Retriever.from_documents(self._all_documents)
            self._bm25.k = initial_k
            return self._bm25
        return None

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        print(f"[*] Adding {len(documents)} chunks to vector database...")
        self.store.add_documents(documents)
        
        # Update local documents and reset BM25 to force re-indexing next time
        if self._all_documents is None:
            self._all_documents = documents
        else:
            self._all_documents.extend(documents)
        self._bm25 = None # Force re-index on next search
        
        print(f"[+] Successfully added all chunks to database")
        return len(documents)

    def search(self, query: str, k: int = 5, filter: Optional[dict] = None, use_reranker: bool = True) -> List[Document]:
        initial_k = k * 4 if use_reranker else k
        
        # 1. Vector Search
        if filter:
            vector_candidates = self.store.similarity_search(query, k=initial_k, filter=filter)
        else:
            vector_candidates = self.store.similarity_search(query, k=initial_k)
        
        # 2. BM25 Search (Keyword matching)
        bm25_candidates = []
        bm25_retriever = self._get_bm25_retriever(initial_k)
        if bm25_retriever:
            bm25_candidates = bm25_retriever.invoke(query)

        # 3. Hybrid Merge (Deduplicate)
        seen_content = set()
        candidates = []
        for doc in (vector_candidates + bm25_candidates):
            if doc.page_content not in seen_content:
                candidates.append(doc)
                seen_content.add(doc.page_content)

        # 4. Reranking stage
        if use_reranker and candidates:
            try:
                from sentence_transformers import CrossEncoder
                if not hasattr(self, '_reranker'):
                    print("[*] Loading Hebrew reranker (DictaBERT-CE)...")
                    self._reranker = CrossEncoder("haguy77/dictabert-ce")
                
                pairs = [[query, doc.page_content] for doc in candidates]
                scores = self._reranker.predict(pairs)
                
                scored_docs = list(zip(candidates, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, score in scored_docs[:k]]
            except Exception as e:
                print(f"[!] Reranker failed: {e}, falling back to hybrid search")
                return candidates[:k]
        
        return candidates[:k]

    def count(self) -> int:
        """Get the number of documents in the store."""
        return self.store._collection.count()

    def clear(self):
        """Clear all documents from the store."""
        # Get all IDs and delete them
        all_ids = self.store._collection.get()['ids']
        if all_ids:
            self.store._collection.delete(ids=all_ids)
        # Recreate the store after clearing
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def exists(self) -> bool:
        """
        Check if the vector store directory exists with data.
        """
        if not os.path.exists(self.persist_directory):
            return False
        # Check if chroma.sqlite3 exists (the actual database file)
        db_file = os.path.join(self.persist_directory, "chroma.sqlite3")
        return os.path.exists(db_file)
