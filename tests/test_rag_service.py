"""
Unit tests for rag_service.py

Mocks retrieval and LLM generation so they do not require Ollama.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rag_service
from config import UNKNOWN_ANSWER


def test_empty_question():
    result = rag_service.ask_question("")
    assert "Please enter a question" in result["answer"]


@patch("rag_service.generate_answer")
@patch("rag_service.retrieve")
def test_refusal_when_no_chunks(mock_retrieve, mock_generate):
    mock_retrieve.return_value = []
    mock_generate.return_value = UNKNOWN_ANSWER
    result = rag_service.ask_question("What is the hostel fee?")
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["refused"] is True
    assert result["sources"] == []


@patch("rag_service.generate_answer")
@patch("rag_service.retrieve")
def test_answer_with_sources(mock_retrieve, mock_generate):
    mock_retrieve.return_value = [
        {"text": "75% attendance required", "source": "academic_regulations.pdf",
         "page": 1, "chunk_index": 0, "distance": 0.2},
    ]
    mock_generate.return_value = "You must maintain 75% attendance."
    result = rag_service.ask_question("What is the attendance requirement?")
    assert result["refused"] is False
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "academic_regulations.pdf"