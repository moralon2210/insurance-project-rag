"""
Document Processor for Hebrew Health Insurance Documents

Orchestrates PDF parsing and text splitting to produce LangChain Document
objects ready for embedding and vector store insertion.

Uses token-based chunking to ensure all content fits within the E5 model's
512 token context window.
"""

import os
import re
from typing import List, Union, Tuple
from langchain_core.documents import Document

from .pdf_parser import PDFParser
from .text_splitter import create_text_splitter
from .tokenizer import E5Tokenizer


class DocumentProcessor:
    """
    Orchestrates the document processing pipeline:
    PDF → Parse → Split → LangChain Documents
    
    Both text and tables are chunked using token-based limits to ensure
    they fit within the embedding model's context window.
    """

    # Token limits for chunking (E5 model has 512 token context window)
    MAX_TEXT_TOKENS = 450      # Leave room for "passage: " prefix
    TEXT_OVERLAP_TOKENS = 50   # Overlap for text chunks
    MAX_TABLE_TOKENS = 400     # Tables need extra room for headers

    def __init__(
        self,
        max_tokens: int = None,
        token_overlap: int = None
    ):
        """
        Initialize the document processor.

        Args:
            max_tokens: Maximum tokens per text chunk (default: 450)
            token_overlap: Token overlap between chunks (default: 50)
        """
        self.max_tokens = max_tokens or self.MAX_TEXT_TOKENS
        self.token_overlap = token_overlap or self.TEXT_OVERLAP_TOKENS
        
        self.parser = PDFParser()
        self.splitter = create_text_splitter(
            max_tokens=self.max_tokens,
            token_overlap=self.token_overlap
        )

    def _extract_tables(self, text: str) -> Tuple[str, List[str], List[int]]:
        """
        Extract Markdown tables from text and replace them with placeholders.
        
        A Markdown table has:
        - Lines starting and ending with |
        - A separator line with dashes (|---|---|)
        
        Args:
            text: The text containing Markdown tables
            
        Returns:
            Tuple of (text_with_placeholders, list_of_tables, list_of_table_positions)
        """
        tables = []
        table_positions = []  # Track where each table starts in the original text
        lines = text.split('\n')
        result_lines = []
        i = 0
        current_char_pos = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line starts a table (starts with | and next line is separator)
            if line.startswith('|') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check if next line is a separator (contains dashes between pipes)
                if re.match(r'^\|[\s\-:]+\|', next_line):
                    # Found table start - save position
                    table_start_pos = current_char_pos
                    
                    # Collect all table lines
                    table_lines = []
                    j = i
                    while j < len(lines) and lines[j].strip().startswith('|'):
                        table_lines.append(lines[j])
                        j += 1
                    
                    # Save table, position, and add placeholder
                    table_text = '\n'.join(table_lines)
                    tables.append(table_text)
                    table_positions.append(table_start_pos)
                    result_lines.append(f'__TABLE_{len(tables) - 1}__')
                    
                    # Update character position
                    for k in range(i, j):
                        current_char_pos += len(lines[k]) + 1  # +1 for newline
                    
                    i = j
                    continue
            
            result_lines.append(lines[i])
            current_char_pos += len(lines[i]) + 1  # +1 for newline
            i += 1
        
        return '\n'.join(result_lines), tables, table_positions

    def _get_page_for_position(self, position: int, page_ranges: List[dict]) -> int:
        """Find which page a character position belongs to."""
        for page_info in page_ranges:
            if page_info["start"] <= position < page_info["end"]:
                return page_info["page"]
        # Default to last page if position is beyond all ranges
        return page_ranges[-1]["page"] if page_ranges else 1

    def _chunk_table(self, table: str, max_tokens: int = None) -> List[str]:
        """
        Split a large table into smaller chunks that fit within token limits.
        
        Each chunk preserves the header row (column names + separator) to maintain
        table structure and context. Rows are not split mid-row.
        
        Args:
            table: The markdown table string
            max_tokens: Maximum tokens per chunk (default: MAX_TABLE_TOKENS)
            
        Returns:
            List of table chunks, each with the header preserved
        """
        max_tokens = max_tokens or self.MAX_TABLE_TOKENS
        
        lines = table.split('\n')
        
        # Need at least header row and separator
        if len(lines) < 2:
            return [table]
        
        # Extract header (first two lines: column names + separator line)
        header = '\n'.join(lines[:2])
        header_tokens = E5Tokenizer.count_tokens(header)
        data_rows = lines[2:]
        
        # If no data rows, return as-is
        if not data_rows:
            return [table]
        
        # If entire table fits, return as-is
        total_tokens = E5Tokenizer.count_tokens(table)
        if total_tokens <= max_tokens:
            return [table]
        
        # Check if header alone exceeds limit (unlikely but handle it)
        if header_tokens >= max_tokens:
            # Truncate the whole table as a fallback
            return [E5Tokenizer.truncate_to_tokens(table, max_tokens)]
        
        # Split table by rows, keeping header in each chunk
        chunks = []
        current_rows = []
        current_tokens = header_tokens
        
        for row in data_rows:
            row_tokens = E5Tokenizer.count_tokens(row)
            
            # If adding this row would exceed limit, save current chunk
            if current_tokens + row_tokens > max_tokens and current_rows:
                chunk_content = header + '\n' + '\n'.join(current_rows)
                chunks.append(chunk_content)
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens
        
        # Add remaining rows as final chunk
        if current_rows:
            chunk_content = header + '\n' + '\n'.join(current_rows)
            chunks.append(chunk_content)
        
        return chunks if chunks else [table]

    def process_file(self, file_path: Union[str]) -> List[Document]:
        """
        Process a single PDF file into LangChain Documents.
        
        Concatenates all pages into one document before chunking to preserve
        tables and content that spans multiple pages.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LangChain Document objects
        """
        # Parse PDF (parser accepts str)
        pages = self.parser.parse_file(file_path)

        if not pages:
            return []

        # Concatenate all pages and track page positions
        full_text = ""
        page_ranges = []  # Track where each page starts/ends
        current_position = 0
        
        for page_data in pages:
            page_content = page_data["content"]
            page_num = page_data["metadata"].get("page", 1)
            
            page_ranges.append({
                "page": page_num,
                "start": current_position,
                "end": current_position + len(page_content)
            })
            
            full_text += page_content + "\n\n"
            current_position += len(page_content) + 2  # +2 for \n\n
        
        full_text = full_text.strip()
        source = pages[0]["metadata"].get("source", "")
        total_pages = len(pages)
        
        # Extract tables as separate chunks
        text_without_tables, tables, table_positions = self._extract_tables(full_text)
        
        # Create document with text only (no tables)
        text_doc = Document(
            page_content=text_without_tables,
            metadata={
                "source": source,
                "total_pages": total_pages
            }
        )
        
        # Split only the text content
        split_docs = self.splitter.split_documents([text_doc])
        
        # Add metadata to text chunks including page info
        for chunk in split_docs:
            chunk.metadata["content_type"] = "text"
            
            # Find chunk position in original text to determine page
            # Use first 50 chars (or less) to find position
            search_text = chunk.page_content[:50] if len(chunk.page_content) >= 50 else chunk.page_content
            # Skip placeholder texts when searching
            if not search_text.startswith("__TABLE_"):
                chunk_start = full_text.find(search_text)
                if chunk_start >= 0:
                    chunk.metadata["page"] = self._get_page_for_position(chunk_start, page_ranges)
        
        # Add tables as chunked documents with page info
        for i, (table, table_start) in enumerate(zip(tables, table_positions)):
            # Find which page this table starts on
            page_num = self._get_page_for_position(table_start, page_ranges)
            
            # Chunk the table if it's too large
            table_chunks = self._chunk_table(table)
            
            for chunk_idx, table_chunk in enumerate(table_chunks):
                table_doc = Document(
                    page_content=table_chunk,
                    metadata={
                        "source": source,
                        "total_pages": total_pages,
                        "content_type": "table",
                        "table_index": i,
                        "table_chunk": chunk_idx + 1,
                        "table_total_chunks": len(table_chunks),
                        "page": page_num
                    }
                )
                split_docs.append(table_doc)
        
        return split_docs

    def process_files(self, file_paths: List[Union[str]]) -> List[Document]:
        """
        Process a list of PDF files.

        Args:
            file_paths: List of paths to PDF files

        Returns:
            List of LangChain Document objects from all PDFs
        """
        all_documents = []

        for file_path in file_paths:
            try:
                docs = self.process_file(file_path)
                all_documents.extend(docs)
                print(f"[+] Processed: {os.path.basename(file_path)} ({len(docs)} chunks)")
            except Exception as e:
                print(f"[!] Error processing {file_path}: {e}")

        return all_documents

    def process_directory(self, directory_path: Union[str]) -> List[Document]:
        """
        Process all PDF files in a directory (including subdirectories).

        Args:
            directory_path: Path to the directory containing PDF files

        Returns:
            List of LangChain Document objects from all PDFs
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Find all PDF files recursively
        pdf_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        if not pdf_files:
            return []

        return self.process_files(pdf_files)
