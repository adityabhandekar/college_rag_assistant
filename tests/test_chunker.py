"""
Unit tests for chunker.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chunker import split_text, chunk_pages


def test_split_text_basic():
    text = "a" * 4000
    chunks = split_text(text, chunk_size=1500, overlap=200)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) <= 1500


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_no_infinite_loop():
    text = "hello world " * 500
    chunks = split_text(text, chunk_size=1500, overlap=200)
    assert len(chunks) > 0


def test_overlap_preserved():
    text = "0123456789" * 500
    chunks = split_text(text, chunk_size=1500, overlap=200)
    # Check that consecutive chunks share content
    for i in range(len(chunks) - 1):
        # Last 200 chars of chunk i should appear at start of chunk i+1
        tail = chunks[i][-200:]
        assert tail in chunks[i + 1] or chunks[i + 1].startswith(tail[:50])


def test_chunk_pages_metadata():
    pages = [
        {"text": "A" * 3000, "source": "test.pdf", "page": 1},
        {"text": "B" * 1000, "source": "test.pdf", "page": 2},
    ]
    chunks = chunk_pages(pages)
    assert all("source" in c for c in chunks)
    assert all("page" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)
    assert chunks[0]["source"] == "test.pdf"


def test_invalid_params():
    try:
        split_text("abc", chunk_size=0, overlap=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    try:
        split_text("abc", chunk_size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass