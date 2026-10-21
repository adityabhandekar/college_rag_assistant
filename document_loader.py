"""
document_loader.py
Reads PDF files from the data directory and extracts text page by page.
Preserves filename and page number metadata.
"""

import logging
import os
from typing import List, Dict

from pypdf import PdfReader

from config import DATA_DIR

logger = logging.getLogger(__name__)


def get_pdf_files(directory: str = DATA_DIR) -> List[str]:
    """
    Return a sorted list of full paths to all .pdf files in the directory.
    """
    if not os.path.isdir(directory):
        logger.warning("Directory does not exist: %s", directory)
        return []

    pdf_files = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(directory, name))

    logger.info("Found %d PDF files in %s", len(pdf_files), directory)
    return pdf_files


def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts:
        {
            "text": str,
            "source": str,    # filename only
            "page": int       # 1-based page number
        }

    Skips empty/unreadable pages without crashing.
    """
    filename = os.path.basename(pdf_path)
    pages = []

    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", filename, exc)
        return pages

    if len(reader.pages) == 0:
        logger.warning("PDF has no pages: %s", filename)
        return pages

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
        except Exception as exc:
            logger.warning(
                "Failed to extract text from %s page %d: %s",
                filename, page_number, exc
            )
            continue

        if page_text is None:
            logger.debug("No text on %s page %d (possibly scanned)", filename, page_number)
            continue

        page_text = page_text.strip()
        if not page_text:
            logger.debug("Empty text on %s page %d", filename, page_number)
            continue

        pages.append({
            "text": page_text,
            "source": filename,
            "page": page_number,
        })

    logger.info("Extracted %d non-empty pages from %s", len(pages), filename)
    return pages


def load_all_documents(directory: str = DATA_DIR) -> List[Dict]:
    """
    Load and extract text from all PDFs in the data directory.

    Returns a flat list of page dicts.
    """
    all_pages = []
    pdf_files = get_pdf_files(directory)

    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return all_pages

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        all_pages.extend(pages)

    logger.info("Total extracted pages across all PDFs: %d", len(all_pages))
    return all_pages