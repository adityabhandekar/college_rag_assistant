"""
config.py
Central configuration for the College RAG Assistant.
All tunable parameters are defined here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1500          # characters
CHUNK_OVERLAP = 200        # characters

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 3                  # number of chunks to retrieve
# Cosine distance threshold. Lower distance = more similar.
# Chunks with distance > this value are considered irrelevant.
# Tune this after testing with your documents.
SIMILARITY_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
COLLECTION_NAME = "college_documents"

# ---------------------------------------------------------------------------
# Refusal message
# ---------------------------------------------------------------------------
UNKNOWN_ANSWER = "Information not found in the available documents."

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")