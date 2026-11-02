"""
DocDeclutter — Metro Document Ingestion Module
Handles loading, text extraction, structural parsing, and metadata cataloging
for transit operational documents (PDF and TXT).
"""

import os
import re
from typing import List, Dict, Any, Optional

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False


class Document:
    """Represents an ingested document with full text, pages, and metadata."""
    def __init__(self, doc_id: str, title: str, doc_type: str, date: str, publisher: str, 
                 content: str, pages: List[Dict[str, Any]], filepath: str):
        self.doc_id = doc_id
        self.title = title
        self.doc_type = doc_type
        self.date = date
        self.publisher = publisher
        self.content = content
        self.pages = pages  # list of {"page_number": int, "text": str}
        self.filepath = filepath

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "date": self.date,
            "publisher": self.publisher,
            "char_count": len(self.content),
            "page_count": len(self.pages),
            "filepath": self.filepath
        }


def extract_metadata_from_text(text: str, default_filename: str) -> Dict[str, str]:
    """Extract metadata (doc_id, title, date, publisher) from document headers."""
    base_id = os.path.splitext(os.path.basename(default_filename))[0]
    metadata = {
        "doc_id": base_id,
        "title": base_id.replace("_", " ").replace("-", " "),
        "doc_type": "Operational Document",
        "date": "2024",
        "publisher": "Kochi Metro Rail Limited (KMRL)"
    }
    
    # Check for Document Code / ID patterns
    code_match = re.search(r"(?:DOCUMENT CODE|DOCUMENT ID|INCIDENT REFERENCE|DOC REF):\s*([A-Za-z0-9\-_/]+)", text, re.IGNORECASE)
    if code_match:
        metadata["doc_id"] = code_match.group(1).strip()
        
    # Check for Title patterns
    title_match = re.search(r"TITLE:\s*([^\n\r=]+)", text, re.IGNORECASE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
        
    # Check for Date patterns
    date_match = re.search(r"(?:DATE OF OCCURRENCE|INSPECTION DATE|DATE):\s*([0-9]{1,2}[-/][A-Za-z0-9]{2,4}[-/][0-9]{2,4}|[0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)
    if date_match:
        metadata["date"] = date_match.group(1).strip()
        
    # Check for Division / Publisher
    pub_match = re.search(r"([A-Za-z\s&]+(?:DIVISION|DEPARTMENT|COMMITTEE|ENGINEERING))", text[:300], re.IGNORECASE)
    if pub_match:
        metadata["publisher"] = pub_match.group(1).strip()
        
    return metadata


def extract_text_from_pdf(filepath: str) -> List[Dict[str, Any]]:
    """Extract text page by page from PDF using PyMuPDF."""
    pages = []
    if not PYMUPDF_AVAILABLE:
        # Fallback reading raw text if fitz is not installed
        try:
            with open(filepath, "rb") as f:
                content = f.read().decode("latin-1", errors="ignore")
                pages.append({"page_number": 1, "text": content})
        except Exception as e:
            pages.append({"page_number": 1, "text": f"[Error reading PDF: {e}]"})
        return pages

    try:
        doc = pymupdf.open(filepath)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text")
            pages.append({
                "page_number": page_idx + 1,
                "text": page_text
            })
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        pages.append({"page_number": 1, "text": f"[Error: {e}]"})
        
    return pages


def extract_text_from_txt(filepath: str) -> List[Dict[str, Any]]:
    """Extract text from plain text file, splitting into simulated pages."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
    except Exception as e:
        full_text = f"[Error reading file: {e}]"

    # Divide text by form-feeds or approximate 2500 character chunks for pages
    raw_pages = full_text.split("\x0c") # Form feed
    if len(raw_pages) <= 1:
        # Approximate pages by line count (~45 lines/page)
        lines = full_text.split("\n")
        chunk_size = 45
        pages_content = ["\n".join(lines[i:i + chunk_size]) for i in range(0, max(len(lines), 1), chunk_size)]
    else:
        pages_content = raw_pages

    pages = []
    for idx, p_text in enumerate(pages_content):
        if p_text.strip():
            pages.append({"page_number": idx + 1, "text": p_text})
            
    if not pages:
        pages.append({"page_number": 1, "text": full_text})
        
    return pages


def ingest_file(filepath: str) -> Optional[Document]:
    """Ingest a single PDF or TXT file into a structured Document object."""
    if not os.path.exists(filepath):
        return None

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        pages = extract_text_from_pdf(filepath)
    elif ext in [".txt", ".log", ".md", ".csv"]:
        pages = extract_text_from_txt(filepath)
    else:
        # Try generic text extraction
        pages = extract_text_from_txt(filepath)

    full_text = "\n\n".join([p["text"] for p in pages])
    meta = extract_metadata_from_text(full_text, filepath)

    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        doc_type=meta["doc_type"],
        date=meta["date"],
        publisher=meta["publisher"],
        content=full_text,
        pages=pages,
        filepath=filepath
    )


def ingest_batch(directory_or_files: Any) -> List[Document]:
    """
    Ingest a batch of documents from a directory path or list of file paths.
    """
    documents = []
    file_list = []

    if isinstance(directory_or_files, str):
        if os.path.isdir(directory_or_files):
            for root, _, files in os.walk(directory_or_files):
                for f in sorted(files):
                    if f.endswith((".pdf", ".txt", ".log", ".md")):
                        file_list.append(os.path.join(root, f))
        elif os.path.isfile(directory_or_files):
            file_list.append(directory_or_files)
    elif isinstance(directory_or_files, list):
        file_list = directory_or_files

    for fpath in file_list:
        doc = ingest_file(fpath)
        if doc and doc.content.strip():
            documents.append(doc)

    return documents
