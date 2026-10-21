"""
rag_service.py
Orchestrates the complete RAG pipeline:
retrieve -> generate -> return answer + sources.
"""

import logging
from typing import Dict, List

from config import UNKNOWN_ANSWER
from retriever import retrieve
from llm_service import generate_answer

logger = logging.getLogger(__name__)


def ask_question(question: str) -> Dict:
    """
    Complete RAG flow.

    Returns:
        {
            "answer": str,
            "sources": [ {"source": str, "page": int, "text": str, "distance": float}, ... ],
            "retrieved": [...],   # same as sources, for display
            "refused": bool
        }
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved": [],
            "refused": False,
        }

    question = question.strip()
    logger.info("Processing question: %s", question)

    # Step 1: Retrieve
    try:
        retrieved = retrieve(question)
    except RuntimeError as exc:
        logger.error("Retrieval error: %s", exc)
        return {
            "answer": f"Retrieval error: {exc}",
            "sources": [],
            "retrieved": [],
            "refused": False,
        }

    # Step 2: Generate
    try:
        answer = generate_answer(question, retrieved)
    except RuntimeError as exc:
        logger.error("Generation error: %s", exc)
        return {
            "answer": f"Generation error: {exc}",
            "sources": [],
            "retrieved": retrieved,
            "refused": False,
        }

    refused = (answer.strip() == UNKNOWN_ANSWER)

    # Build sources list (deduplicated by source+page)
    sources = []
    seen = set()
    for chunk in retrieved:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": chunk["distance"],
            })

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": retrieved,
        "refused": refused,
    }