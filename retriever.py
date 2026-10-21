"""
retriever.py
Semantic retrieval from ChromaDB.
Returns top-K chunks with source metadata, filtered by a distance threshold.
"""

import logging
from typing import List, Dict

from config import TOP_K, SIMILARITY_THRESHOLD
from embeddings import embed_query
from vector_store import get_client, get_collection

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.

    Returns a list of dicts:
        {
            "text": str,
            "source": str,
            "page": int,
            "chunk_index": int,
            "distance": float
        }

    Only chunks with distance <= threshold are returned.
    """
    if not query or not query.strip():
        logger.warning("Empty query received.")
        return []

    try:
        query_vector = embed_query(query)
    except RuntimeError as exc:
        logger.error("Query embedding failed: %s", exc)
        raise

    try:
        client = get_client()
        collection = get_collection(client)
    except Exception as exc:
        logger.error("Could not access vector store: %s", exc)
        raise RuntimeError(f"Vector store unavailable: {exc}") from exc

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    # Parse results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist is None:
            continue
        if dist > threshold:
            logger.debug("Skipping chunk with distance %.4f (> threshold %.4f)", dist, threshold)
            continue
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })

    logger.info("Retrieved %d chunks above threshold for query: %s", len(retrieved), query[:50])
    return retrieved