"""
Text Splitter for Hebrew Health Insurance Documents

Provides Hebrew-optimized text chunking for RAG pipelines.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional


def create_text_splitter(
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter optimized for Hebrew insurance documents.

    Args:
        chunk_size: Maximum size of each chunk (default: 600 characters)
        chunk_overlap: Overlap between consecutive chunks (default: 120 characters)
        separators: Custom separators list (default: Hebrew-optimized separators)

    Returns:
        Configured RecursiveCharacterTextSplitter instance
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
