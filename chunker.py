"""
chunker.py
Splits extracted page text into overlapping chunks.
Preserves source filename and page metadata.
"""

import logging
from typing import List, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a single string into overlapping chunks.

    - chunk_size: maximum number of characters per chunk
    - overlap: number of characters shared between consecutive chunks

    Guards against empty text and infinite loops.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Convert a list of page dicts into a list of chunk dicts.

    Each output dict:
        {
            "text": str,
            "source": str,
            "page": int,
            "chunk_index": int   # index within the page (0-based)
        }
    """
    all_chunks = []

    for page in pages:
        page_text = page.get("text", "")
        source = page.get("source", "unknown")
        page_number = page.get("page", 0)

        text_chunks = split_text(page_text)

        for idx, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page_number,
                "chunk_index": idx,
            })

    logger.info("Created %d chunks from %d pages", len(all_chunks), len(pages))
    return all_chunks