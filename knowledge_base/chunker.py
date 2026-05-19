"""
chunker.py — Document Loading & Text Chunking

Loads raw medical documents (PDF and TXT) from a directory and splits them
into semantically meaningful chunks suitable for embedding.

Chunking Strategy:
    - Chunk size: ~500 tokens (~2000 characters)
    - Overlap: ~50 tokens (~200 characters)
    - Splitter: RecursiveCharacterTextSplitter (splits on paragraphs → 
      sentences → words, preserving semantic boundaries)

Each chunk is returned with metadata tracking its source file and position,
so we can trace any retrieved chunk back to its original document.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of text with provenance metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: source_file, chunk_index, total_chunks


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------

def load_text_file(file_path: str) -> str:
    """Load a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """
    Load a PDF file and extract text from all pages.
    
    Concatenates text from each page with double newlines to preserve
    logical separation between pages.
    """
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_document(file_path: str) -> str:
    """
    Load a document based on its file extension.
    
    Supported formats:
        - .txt  → plain text
        - .pdf  → PDF extraction
    
    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == ".txt":
        return load_text_file(file_path)
    elif ext == ".pdf":
        return load_pdf_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: .txt, .pdf"
        )


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a document's text into overlapping chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries in this order: paragraphs (\\n\\n) → newlines (\\n) → 
    sentences (. ) → spaces ( ) → characters.
    
    This preserves semantic coherence within each chunk — similar to how
    you'd want to split a graph at natural cut vertices rather than 
    arbitrarily slicing edges.
    
    Args:
        text:           The full document text.
        source_file:    Path to the original file (stored in metadata).
        chunk_size:     Target chunk size in characters (~500 tokens).
        chunk_overlap:  Overlap between consecutive chunks in characters.
    
    Returns:
        List of DocumentChunk objects with text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splits = splitter.split_text(text)
    
    chunks = []
    for i, split_text in enumerate(splits):
        chunk = DocumentChunk(
            text=split_text.strip(),
            metadata={
                "source_file": os.path.basename(source_file),
                "chunk_index": i,
                "total_chunks": len(splits),
            },
        )
        chunks.append(chunk)
    
    return chunks


# ---------------------------------------------------------------------------
# Pipeline: Load & Chunk an Entire Directory
# ---------------------------------------------------------------------------

def load_and_chunk_directory(
    directory: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Load all supported documents from a directory and chunk them.
    
    Walks through the directory (non-recursive), loads each .txt and .pdf
    file, and splits them into chunks. Think of this as the "preprocessing 
    pipeline" that transforms raw data into the atomic units our embedding 
    model will operate on.
    
    Args:
        directory:      Path to the directory containing documents.
        chunk_size:     Target chunk size in characters.
        chunk_overlap:  Overlap between consecutive chunks.
    
    Returns:
        List of all DocumentChunk objects across all files.
    
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    supported_extensions = {".txt", ".pdf"}
    all_chunks = []
    files_processed = 0
    
    # Sort for deterministic ordering across runs
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in supported_extensions:
            print(f"  📄 Loading: {file_path.name}")
            
            text = load_document(str(file_path))
            
            if not text.strip():
                print(f"  ⚠️  Skipping empty file: {file_path.name}")
                continue
            
            chunks = chunk_text(
                text=text,
                source_file=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            all_chunks.extend(chunks)
            files_processed += 1
            print(f"     → {len(chunks)} chunks created")
    
    print(f"\n  ✅ Total: {files_processed} files → {len(all_chunks)} chunks")
    return all_chunks
