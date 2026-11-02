"""
DocDeclutter — Chunking Engine
Implements Structure-Aware Chunking (Novelty 1) and Baseline Fixed-Size Chunking
for transit operational documents.
"""

import re
from typing import List, Dict, Any
from src.ingestion import Document


class Chunk:
    """Represents a chunk of text with rich structural metadata."""
    def __init__(self, chunk_id: str, doc_id: str, doc_title: str, section_title: str,
                 page_number: int, text: str, chunking_method: str = "structure_aware"):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.section_title = section_title
        self.page_number = page_number
        self.text = text
        self.chunking_method = chunking_method

    @property
    def contextual_text(self) -> str:
        """Text prefixed with traceable structural context."""
        return (f"[Document: {self.doc_id} | Title: {self.doc_title} | "
                f"Section: {self.section_title} | Page: {self.page_number}]\n"
                f"{self.text}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "text": self.text,
            "contextual_text": self.contextual_text,
            "chunking_method": self.chunking_method,
            "char_count": len(self.text)
        }


def structure_aware_chunking(document: Document, max_chunk_chars: int = 1200, 
                             overlap_chars: int = 150) -> List[Chunk]:
    """
    Structure-Aware Chunking (Novelty 1):
    Splits text by identifying section headings, subsections, list items, and page boundaries,
    preserving structural hierarchy and metadata traceability.
    """
    chunks = []
    chunk_counter = 1

    # Regex pattern to match Section headers
    section_pattern = re.compile(
        r"(SECTION\s+[0-9]+(?:\.[0-9]+)*[:\-]?\s*[^\n\r]+|"
        r"^[0-9]+\.[0-9]+\s+[^\n\r]+|"
        r"^[A-Z\s]{4,30}:(?:\s*[\r\n]|$)|"
        r"^===+\s*[\r\n])",
        re.MULTILINE
    )

    for page in document.pages:
        page_num = page["page_number"]
        page_text = page["text"]
        
        # Split text into sections based on section headers
        splits = list(section_pattern.finditer(page_text))
        
        if not splits:
            # No explicit section header found on this page; use paragraph boundaries
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            current_section = f"General Content (Page {page_num})"
            
            curr_text = ""
            for p in paragraphs:
                if len(curr_text) + len(p) + 2 > max_chunk_chars and curr_text:
                    cid = f"{document.doc_id}_C{chunk_counter:03d}"
                    chunks.append(Chunk(
                        chunk_id=cid,
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        section_title=current_section,
                        page_number=page_num,
                        text=curr_text.strip(),
                        chunking_method="structure_aware"
                    ))
                    chunk_counter += 1
                    # Keep overlap
                    curr_text = curr_text[-overlap_chars:] + "\n" + p
                else:
                    curr_text = curr_text + "\n\n" + p if curr_text else p
                    
            if curr_text.strip():
                cid = f"{document.doc_id}_C{chunk_counter:03d}"
                chunks.append(Chunk(
                    chunk_id=cid,
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    section_title=current_section,
                    page_number=page_num,
                    text=curr_text.strip(),
                    chunking_method="structure_aware"
                ))
                chunk_counter += 1
        else:
            # We have identified section headers
            current_section_title = f"Document Introduction (Page {page_num})"
            last_pos = 0
            
            for idx, match in enumerate(splits):
                start, end = match.span()
                # Text before this match
                preceding_text = page_text[last_pos:start].strip()
                if preceding_text:
                    # Append chunk for the preceding section
                    cid = f"{document.doc_id}_C{chunk_counter:03d}"
                    chunks.append(Chunk(
                        chunk_id=cid,
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        section_title=current_section_title,
                        page_number=page_num,
                        text=preceding_text,
                        chunking_method="structure_aware"
                    ))
                    chunk_counter += 1
                    
                matched_header = match.group(0).strip().replace("=", "").strip()
                if matched_header:
                    current_section_title = matched_header
                last_pos = end

            # Trailing text after the last header
            trailing_text = page_text[last_pos:].strip()
            if trailing_text:
                cid = f"{document.doc_id}_C{chunk_counter:03d}"
                chunks.append(Chunk(
                    chunk_id=cid,
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    section_title=current_section_title,
                    page_number=page_num,
                    text=trailing_text,
                    chunking_method="structure_aware"
                ))
                chunk_counter += 1

    return chunks


def baseline_fixed_chunking(document: Document, chunk_size: int = 600, 
                           stride: int = 500) -> List[Chunk]:
    """
    Baseline Fixed-Size Chunking (Comparative Baseline):
    Splits text blindly into fixed-character windows with no awareness
    of section headers, tables, or document structure.
    """
    chunks = []
    raw_text = document.content
    chunk_counter = 1
    
    start = 0
    total_len = len(raw_text)
    
    while start < total_len:
        end = min(start + chunk_size, total_len)
        segment = raw_text[start:end].strip()
        
        # Estimate page by position in total text
        approx_page = max(1, int((start / total_len) * len(document.pages)) + 1)
        
        if segment:
            cid = f"{document.doc_id}_BASE_{chunk_counter:03d}"
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=document.doc_id,
                doc_title=document.title,
                section_title="Unknown Section (Baseline)",
                page_number=approx_page,
                text=segment,
                chunking_method="baseline_fixed"
            ))
            chunk_counter += 1
            
        start += stride

    return chunks


def chunk_documents_batch(documents: List[Document], method: str = "structure_aware") -> List[Chunk]:
    """Chunk a list of documents using the requested method."""
    all_chunks = []
    for doc in documents:
        if method == "structure_aware":
            doc_chunks = structure_aware_chunking(doc)
        else:
            doc_chunks = baseline_fixed_chunking(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks

