"""
vector_store.py
Manages the ChromaDB persistent vector store.
Stores chunk text, embeddings, source filename, page number, and chunk index.
Prevents duplicate entries on re-indexing.
"""

import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_DIR, COLLECTION_NAME
from embeddings import embed_texts

logger = logging.getLogger(__name__)


def get_client() -> chromadb.PersistentClient:
    """
    Return a persistent ChromaDB client.
    Data is stored on disk and loaded on startup.
    """
    return chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: Optional[chromadb.PersistentClient] = None):
    """
    Get or create the document collection.
    """
    if client is None:
        client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def clear_collection(client: Optional[chromadb.PersistentClient] = None):
    """
    Delete all documents from the collection.
    Used before re-indexing.
    """
    if client is None:
        client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted collection '%s'.", COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Could not delete collection (may not exist): %s", exc)
    # Recreate empty collection
    client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(chunks: List[Dict], batch_size: int = 50) -> int:
    """
    Add chunks to the vector store.

    Each chunk dict must contain:
        text, source, page, chunk_index

    Returns the number of chunks added.
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return 0

    client = get_client()
    collection = get_collection(client)

    total_added = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        ids = []
        documents = []
        metadatas = []

        for chunk in batch:
            source = chunk["source"]
            page = chunk["page"]
            chunk_index = chunk["chunk_index"]
            # Unique ID per chunk
            uid = f"{source}::p{page}::c{chunk_index}"
            ids.append(uid)
            documents.append(chunk["text"])
            metadatas.append({
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
            })

        texts = [c["text"] for c in batch]
        try:
            embeddings = embed_texts(texts)
        except RuntimeError as exc:
            logger.error("Embedding failed during indexing: %s", exc)
            raise

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_added += len(batch)
        logger.info("Added batch of %d chunks (total %d).", len(batch), total_added)

    return total_added


def get_collection_count() -> int:
    """
    Return the number of documents in the collection.
    """
    try:
        client = get_client()
        collection = get_collection(client)
        return collection.count()
    except Exception as exc:
        logger.warning("Could not get collection count: %s", exc)
        return 0


def get_all_sources() -> List[str]:
    """
    Return a sorted list of unique source filenames in the collection.
    """
    try:
        client = get_client()
        collection = get_collection(client)
        result = collection.get(include=["metadatas"])
        sources = set()
        for meta in result.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
    except Exception as exc:
        logger.warning("Could not list sources: %s", exc)
        return []