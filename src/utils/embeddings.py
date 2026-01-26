# -*- coding: utf-8 -*-
"""
Custom Embedding Models for Hebrew Health Insurance RAG System

Implements E5 embeddings with proper query/passage prefixes for optimal performance.
"""

from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class E5Embeddings(Embeddings):
    """
    Custom E5 embedding class using sentence-transformers.
    
    E5 models are designed to work with prefixes:
    - "query: " for search queries
    - "passage: " for documents
    
    This provides better semantic understanding and retrieval performance.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        """
        Initialize E5 embeddings.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
            normalize_embeddings: Whether to normalize embeddings (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        # Load the model
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with "passage: " prefix.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embeddings as lists of floats
        """
        # Add passage prefix for documents
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_tensor=False,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query with "query: " prefix.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding as a list of floats
        """
        # Add query prefix for search queries
        prefixed_text = f"query: {text}"
        
        # Generate embedding
        embedding = self.model.encode(
            prefixed_text,
            convert_to_tensor=False,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        
        return embedding.tolist()
