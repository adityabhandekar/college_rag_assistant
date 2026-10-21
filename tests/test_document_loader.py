"""
Unit tests for document_loader.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_loader import get_pdf_files, extract_pages, load_all_documents


def test_get_pdf_files_missing_dir():
    result = get_pdf_files("nonexistent_directory_xyz")
    assert result == []


def test_extract_pages_missing_file():
    result = extract_pages("nonexistent_file.pdf")
    assert result == []


def test_load_all_documents_missing_dir():
    result = load_all_documents("nonexistent_directory_xyz")
    assert result == []