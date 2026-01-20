"""
Document Processor for Hebrew Health Insurance Documents

Orchestrates PDF parsing and text splitting to produce LangChain Document
objects ready for embedding and vector store insertion.
"""

import os
import re
from typing import List, Union, Tuple
from langchain_core.documents import Document

from .pdf_parser import PDFParser
from .text_splitter import create_text_splitter


class DocumentProcessor:
    """
    Orchestrates the document processing pipeline:
    PDF → Parse → Split → LangChain Documents
    
    Tables are extracted and kept as whole chunks to preserve their structure.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.parser = PDFParser()
        self.splitter = create_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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
        
        # Add tables as separate whole chunks with page info
        for i, (table, table_start) in enumerate(zip(tables, table_positions)):
            # Find which page this table starts on
            page_num = self._get_page_for_position(table_start, page_ranges)
            
            table_doc = Document(
                page_content=table,
                metadata={
                    "source": source,
                    "total_pages": total_pages,
                    "content_type": "table",
                    "table_index": i,
                    "page": page_num
                }
            )
            split_docs.append(table_doc)
        
        return split_docs

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
                print(f"Processed: {os.path.basename(file_path)} ({len(docs)} chunks)")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return all_documents
