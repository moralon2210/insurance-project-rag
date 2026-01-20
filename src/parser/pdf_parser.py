"""
PDF Parser for Hebrew Health Insurance Documents

Extracts text and tables from PDF files, converting tables to Markdown format
for better LLM comprehension in RAG pipelines.
"""

import pdfplumber
import os
import re
from typing import List, Dict, Any, Optional


class PDFParser:
    """
    Unified PDF parser that extracts text and tables from Hebrew insurance documents.
    Tables are converted to Markdown format to preserve structure.
    """

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and extract content from all pages.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of dictionaries containing page content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        pages_content = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                content = self._extract_page_content(page)
                pages_content.append({
                    "content": content,
                    "metadata": {
                        "source": file_path,
                        "page": page_num,
                        "total_pages": len(pdf.pages)
                    }
                })

        return pages_content

    def _extract_page_content(self, page) -> str:
        """
        Extract content from a single page, handling both text and tables.

        Args:
            page: pdfplumber page object

        Returns:
            Combined content as string with tables in Markdown format
        """
        # Get tables from the page
        tables = page.extract_tables()

        if not tables:
            # No tables - use character-level extraction for Hebrew
            text = self._extract_text_from_chars(page)
            return text.strip()

        # Get table objects with bounding boxes to maintain reading order
        table_objects = page.find_tables()
        
        # Create list of (y_position, content_type, content)
        page_elements = []
        
        # Add tables with their positions
        for table_obj, table_data in zip(table_objects, tables):
            md_table = self._table_to_markdown(table_data)
            if md_table:
                # Use top of bounding box for position
                y_position = table_obj.bbox[1]
                page_elements.append((y_position, "table", md_table))
        
        # Extract text outside tables
        table_bboxes = [t.bbox for t in table_objects]
        text_outside = self._extract_text_outside_tables(page, table_bboxes)
        
        if text_outside:
            # Put text at position 0 (top of page)
            page_elements.append((0, "text", text_outside))
        
        # Sort by y-position to maintain reading order
        page_elements.sort(key=lambda x: x[0])
        
        # Extract only the content
        content_parts = [element[2] for element in page_elements]
        
        return "\n\n".join(content_parts)

    def _extract_text_outside_tables(self, page, table_bboxes: List[tuple]) -> str:
        """
        Extract text from page areas that are not covered by tables.

        Args:
            page: pdfplumber page object
            table_bboxes: List of table bounding boxes (x0, y0, x1, y1)

        Returns:
            Text content from non-table areas
        """
        return self._extract_text_from_chars(page, exclude_bboxes=table_bboxes)

    def _is_within_bbox(self, obj: Dict, bbox: tuple) -> bool:
        """
        Check if a text object is within a bounding box.

        Args:
            obj: pdfplumber character/text object
            bbox: Bounding box (x0, y0, x1, y1)

        Returns:
            True if object is within the bbox
        """
        if "x0" not in obj or "top" not in obj:
            return False

        x0, y0, x1, y1 = bbox
        obj_x = obj.get("x0", 0)
        obj_y = obj.get("top", 0)

        return x0 <= obj_x <= x1 and y0 <= obj_y <= y1

    def _extract_text_from_chars(self, page, exclude_bboxes: List[tuple] = None) -> str:
        """
        Extract text using character-level extraction to avoid duplication.
        
        This method extracts each character individually and reconstructs the text,
        which avoids the duplication issues that occur with extract_text() for Hebrew.
        It also handles PDFs that store characters multiple times at the same position
        (a technique used for bold/thick text effects).
        
        Args:
            page: pdfplumber page object
            exclude_bboxes: Optional list of bounding boxes to exclude (e.g., tables)
            
        Returns:
            Extracted text as a string
        """
        chars = page.chars
        
        if not chars:
            return ""
        
        # Filter out chars inside excluded bboxes (like tables)
        if exclude_bboxes:
            chars = [c for c in chars if not any(
                self._is_within_bbox(c, bbox) for bbox in exclude_bboxes
            )]
        
        if not chars:
            return ""
        
        # Deduplicate characters at the same position
        # Some PDFs store the same character multiple times at nearly identical positions
        # (used for bold/thick text effects)
        POSITION_TOLERANCE = 2  # pixels
        deduplicated_chars = []
        seen_positions = set()
        
        for char in chars:
            # Round position to tolerance
            pos_key = (
                round(char['x0'] / POSITION_TOLERANCE) * POSITION_TOLERANCE,
                round(char['top'] / POSITION_TOLERANCE) * POSITION_TOLERANCE,
                char['text']
            )
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                deduplicated_chars.append(char)
        
        chars = deduplicated_chars
        
        if not chars:
            return ""
        
        # Sort chars by y position (top to bottom)
        chars = sorted(chars, key=lambda c: c['top'])
        
        # Group characters into lines based on y-coordinate proximity
        LINE_TOLERANCE = 5  # pixels (increased to handle slight vertical variations)
        lines = []
        current_line = [chars[0]]
        
        for char in chars[1:]:
            if abs(char['top'] - current_line[0]['top']) <= LINE_TOLERANCE:
                current_line.append(char)
            else:
                lines.append(current_line)
                current_line = [char]
        
        # Don't forget the last line
        if current_line:
            lines.append(current_line)
        
        # Build text from lines
        text_lines = []
        for line in lines:
            # Sort chars within line by x position (left to right)
            # Note: For Hebrew RTL, the PDF already stores chars in visual order,
            # so we sort left-to-right and let the display handle RTL
            line_chars = sorted(line, key=lambda c: c['x0'])
            line_text = ''.join(c['text'] for c in line_chars)
            # Apply text deduplication for any remaining duplicates
            line_text = self._deduplicate_text(line_text)
            text_lines.append(line_text)
        
        return '\n'.join(text_lines)

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """
        Convert a table (list of rows) to Markdown format.

        Args:
            table: List of rows, where each row is a list of cell values

        Returns:
            Markdown formatted table string
        """
        if not table:
            return ""

        # Clean up cells - replace None with empty string
        cleaned_table = []
        for row in table:
            if row:
                cleaned_row = [self._clean_cell(cell) for cell in row]
                cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        # Determine column count from the first row
        col_count = len(cleaned_table[0])
        if col_count == 0:
            return ""

        # Build Markdown table
        lines = []

        # Header row (first row of table)
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator row
        lines.append("| " + " | ".join(["---"] * col_count) + " |")

        # Data rows
        for row in cleaned_table[1:]:
            # Ensure row has correct number of columns
            while len(row) < col_count:
                row.append("")
            lines.append("| " + " | ".join(row[:col_count]) + " |")

        return "\n".join(lines)

    def _clean_cell(self, cell: Optional[str]) -> str:
        """
        Clean a table cell value.

        Args:
            cell: Cell value (may be None)

        Returns:
            Cleaned string value
        """
        if cell is None:
            return ""

        # Convert to string and clean up
        text = str(cell).strip()

        # Replace newlines with spaces (cells shouldn't have line breaks in Markdown)
        text = text.replace("\n", " ").replace("\r", "")
        
        # Deduplicate repeated characters (caused by PDF bold/thick text effects)
        # Pattern: same character repeated 4+ times in a row -> single character
        text = self._deduplicate_text(text)

        # Escape pipe characters as they're Markdown table delimiters
        text = text.replace("|", "\\|")

        return text
    
    def _deduplicate_text(self, text: str) -> str:
        """
        Remove duplicate characters caused by PDF bold/thick text effects.
        
        Some PDFs store characters multiple times at the same position,
        which results in text like "הההה" instead of "ה". This function
        removes such duplications.
        
        The PDF bold effect typically duplicates each character exactly 4 times,
        so we only remove sequences of 4+ identical characters to avoid
        breaking legitimate repeated characters like "000" in "2,000,000".
        
        Args:
            text: Text that may contain duplicated characters
            
        Returns:
            Text with duplications removed
        """
        if not text:
            return text
        
        # Deduplicate Hebrew characters (2+ repeats -> single)
        # Only applies to Hebrew Unicode range to preserve numbers like "000"
        result = re.sub(r'([\u0590-\u05FF])\1{1,}', r'\1', text)
        
        # For digits and other chars, only deduplicate 4+ repeats (bold effect)
        result = re.sub(r'([^\u0590-\u05FF])\1{3,}', r'\1', result)
        
        return result
