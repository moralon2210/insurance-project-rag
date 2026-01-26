# -*- coding: utf-8 -*-
"""
Tokenizer utility for Hebrew Health Insurance RAG System

Provides accurate token counting using the E5 model tokenizer.
This ensures chunks fit within the embedding model's context window.
"""

from typing import Optional
from transformers import AutoTokenizer


class E5Tokenizer:
    """
    Singleton tokenizer for E5 model token counting.
    
    E5 multilingual model has a 512 token context window.
    This class provides utilities to count tokens and validate chunk sizes.
    """
    
    _instance: Optional[AutoTokenizer] = None
    MODEL_NAME = "intfloat/multilingual-e5-base"
    MAX_TOKENS = 512
    
    @classmethod
    def get_instance(cls) -> AutoTokenizer:
        """Get or create the tokenizer instance (lazy loading)."""
        if cls._instance is None:
            print("[*] Loading E5 tokenizer...")
            cls._instance = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            print("[+] E5 tokenizer loaded")
        return cls._instance
    
    @classmethod
    def count_tokens(cls, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Number of tokens (including special tokens)
        """
        if not text:
            return 0
        tokenizer = cls.get_instance()
        return len(tokenizer.encode(text, add_special_tokens=True))
    
    @classmethod
    def fits_context(cls, text: str, max_tokens: Optional[int] = None) -> bool:
        """
        Check if a text fits within the context window.
        
        Args:
            text: The text to check
            max_tokens: Maximum allowed tokens (default: 512)
            
        Returns:
            True if text fits, False otherwise
        """
        max_tokens = max_tokens or cls.MAX_TOKENS
        return cls.count_tokens(text) <= max_tokens
    
    @classmethod
    def truncate_to_tokens(cls, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token limit.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated text that fits within the token limit
        """
        tokenizer = cls.get_instance()
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
