# College Document Q&A Assistant (Local RAG)

A locally hosted Retrieval-Augmented Generation (RAG) assistant that answers
student questions from college PDF documents and refuses to invent information
when the answer is not present.

**Everything runs on your computer. No OpenAI, Gemini, Claude, or paid APIs.**

---

## Features

- Reads multiple college PDFs and extracts text page by page.
- Splits text into ~1,500-character chunks with 200-character overlap.
- Generates embeddings with Ollama `nomic-embed-text`.
- Stores embeddings persistently in ChromaDB.
- Retrieves top-3 relevant chunks using semantic similarity.
- Answers with Ollama `llama3.2:3b` using only retrieved context.
- Displays source filename, page number, and retrieved excerpts.
- Returns **"Information not found in the available documents."** when
  evidence is insufficient.
- Streamlit UI with re-index button, index status, and error handling.

---

## Prerequisites

- Windows 10/11 (also works on macOS/Linux)
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- At least 8 GB RAM (for `llama3.2:3b`)

---

## Installation (Windows PowerShell)

```powershell
cd college-rag-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt