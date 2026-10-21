"""
embeddings.py
Generates embeddings using Ollama's nomic-embed-text model.
"""

import logging
from typing import List

import ollama

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Returns a list of embedding vectors (each a list of floats).
    Raises RuntimeError if the Ollama service is unavailable or returns an error.
    """
    if not texts:
        return []

    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except Exception as exc:
        logger.error("Ollama embedding call failed: %s", exc)
        raise RuntimeError(
            f"Could not generate embeddings. Is Ollama running and is '{EMBEDDING_MODEL}' pulled? "
            f"Original error: {exc}"
        ) from exc

    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, got {len(embeddings)}."
        )

    return embeddings


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    """
    result = embed_texts([query])
    return result[0]