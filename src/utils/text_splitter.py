"""
Text Splitter for Hebrew Health Insurance Documents

Provides Hebrew-optimized text chunking for RAG pipelines.
Uses token-based chunking to ensure chunks fit within E5 model context window (512 tokens).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional

from .tokenizer import E5Tokenizer


def create_text_splitter(
    max_tokens: int = 450,
    token_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter optimized for Hebrew insurance documents.
    
    Uses token-based chunking to ensure chunks fit within the E5 model's
    512 token context window. Default max_tokens=450 leaves room for
    the "passage: " prefix added during embedding.

    Args:
        max_tokens: Maximum tokens per chunk (default: 450)
        token_overlap: Overlap in tokens between chunks (default: 50)
        separators: Custom separators list (default: Hebrew-optimized separators)

    Returns:
        Configured RecursiveCharacterTextSplitter instance with token-based length function
    """
    if separators is None:
        # Hebrew-optimized separators
        separators = [
            "\n\n",  # Paragraph breaks - highest priority
            "\n",    # Line breaks
            ".",     # Sentence endings
            ":",     # Common in insurance docs (e.g., "כיסוי:")
            " ",     # Word boundaries
            "",      # Character by character (fallback)
        ]

    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=token_overlap,
        separators=separators,
        length_function=E5Tokenizer.count_tokens,  # Token-based length!
        is_separator_regex=False,
    )
