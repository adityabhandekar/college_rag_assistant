"""
Unit tests for retriever.py

These tests mock the embedding and vector store layers so they do not
require a running Ollama instance.
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import retriever


def test_retrieve_empty_query():
    result = retriever.retrieve("")
    assert result == []


@patch("retriever.get_collection")
@patch("retriever.get_client")
@patch("retriever.embed_query")
def test_retrieve_filters_by_threshold(mock_embed, mock_client, mock_collection):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection_obj = MagicMock()
    mock_collection_obj.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"source": "a.pdf", "page": 1, "chunk_index": 0},
                       {"source": "b.pdf", "page": 2, "chunk_index": 1}]],
        "distances": [[0.3, 0.9]],
    }
    mock_collection.return_value = mock_collection_obj
    mock_client.return_value = MagicMock()

    result = retriever.retrieve("test query", top_k=3, threshold=0.75)
    assert len(result) == 1
    assert result[0]["source"] == "a.pdf"


@patch("retriever.get_collection")
@patch("retriever.get_client")
@patch("retriever.embed_query")
def test_retrieve_no_results(mock_embed, mock_client, mock_collection):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection_obj = MagicMock()
    mock_collection_obj.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_collection.return_value = mock_collection_obj
    mock_client.return_value = MagicMock()

    result = retriever.retrieve("test query")
    assert result == []