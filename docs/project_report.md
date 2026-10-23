# Project Report

## Design and Development of a Local LLM-Based College Document Question Answering Assistant Using Retrieval-Augmented Generation (RAG)

---

## Abstract

Students frequently need to find information buried across many college PDF documents — academic regulations, examination guidelines, placement criteria, fee rules, and handbooks. Manual search is slow and error-prone. This project builds a locally hosted question-answering assistant using Retrieval-Augmented Generation (RAG). PDF text is extracted, chunked, embedded with `nomic-embed-text`, and stored in ChromaDB. A user question is embedded, the top-3 most similar chunks are retrieved, and a local LLM (`llama3.2:3b` via Ollama) generates an answer using **only** the retrieved context. The system displays source filenames and page numbers, and refuses to answer when evidence is insufficient, replying exactly: **"Information not found in the available documents."** Everything runs on the user's computer without paid APIs. A project management component is tracked in Jira Free.

---

## 1. Introduction

Colleges maintain large volumes of policy and academic information in PDFs. Students repeatedly need the same answers (attendance requirements, credit rules, examination procedures, fee deadlines). Keyword search misses paraphrases; general-purpose LLMs may hallucinate. A local RAG system addresses both problems by retrieving relevant passages from official documents and constraining the LLM to those passages.

## 2. Problem Statement

Students must manually search multiple PDFs to find policy answers. Keyword search fails on paraphrased questions (e.g., "How many classes can I miss?" vs. "minimum attendance of 75%"). General LLMs may invent fees, dates, or rules. The system must answer from the documents and refuse when it cannot.

## 3. Objectives

1. Read multiple college PDF documents.
2. Extract text with page metadata.
3. Chunk text with configurable size and overlap.
4. Generate embeddings using a local model.
5. Store embeddings in a persistent local vector database.
6. Accept natural-language questions.
7. Retrieve top-3 relevant chunks.
8. Generate answers using only retrieved context.
9. Display source filenames and pages.
10. Refuse to invent information.
11. Support re-indexing when documents change.
12. Provide a usable Streamlit UI.
13. Evaluate retrieval accuracy and safe refusal rate.

## 4. Scope

**In scope:**
- Multiple PDFs, text extraction, chunking, embeddings, retrieval, LLM answer generation, source citation, refusal logic, Streamlit UI, Jira tracking, evaluation dataset.

**Out of scope:**
- OCR for scanned PDFs.
- User authentication.
- Multi-language UI.
- Cloud hosting.
- Real-time document updates.

## 5. Existing System

- Manual PDF search or Ctrl+F.
- Keyword-based document search.
- Direct use of general LLMs (hallucination risk).

**Limitations:** slow, misses paraphrases, cannot cite sources, may hallucinate.

## 6. Proposed System

A modular local RAG application:

- **Ingestion pipeline:** PDF extraction → chunking → embedding → ChromaDB.
- **Query pipeline:** embed question → retrieve top-3 → threshold filter → prompt → Ollama chat → answer + sources.
- **UI:** Streamlit with re-index button, index status, expandable excerpts.
- **Safety:** empty-retrieval shortcut + strict prompt + configurable threshold.

## 7. Methodology

1. Requirements and dataset collection.
2. Local LLM setup (Ollama + models).
3. RAG pipeline build (modules in the defined order).
4. Streamlit UI.
5. Testing and evaluation (50 labelled questions).
6. Documentation and demo.

Four one-week sprints in Jira (Foundation, RAG Development, Complete LLM Application, Testing and Submission).

## 8. Architecture

See `docs/architecture.md`. Summary:

- `document_loader.py` → `chunker.py` → `embeddings.py` → `vector_store.py`
- `retriever.py` → `llm_service.py` → `rag_service.py` → `app.py`

## 9. Technology Stack

| Component | Technology | Cost |
|---|---|---|
| Language | Python 3.10+ | Free |
| Local LLM runtime | Ollama | Free |
| Chat model | `llama3.2:3b` | Free |
| Embedding model | `nomic-embed-text` | Free |
| Vector DB | ChromaDB (persistent) | Free |
| PDF processing | PyPDF | Free |
| UI | Streamlit | Free |
| Project management | Jira Free | Free |
| IDE | VS Code | Free |
| Hosting | Student computer | Free |
| OpenAI / Gemini / Claude | Not used | — |

## 10. Implementation

### 10.1 PDF extraction
`pypdf.PdfReader` reads each page. Empty/unreadable pages are skipped and logged. Filename and 1-based page number are preserved.

### 10.2 Chunking
Text is split into 1,500-character chunks with 200-character overlap. Guards prevent infinite loops and empty chunks. Each chunk carries `source`, `page`, and `chunk_index`.

### 10.3 Embeddings
Ollama `nomic-embed-text` is called via `ollama.embed()` in batches. Runtime errors (service down, model missing) raise clear `RuntimeError`s.

### 10.4 Vector storage
ChromaDB `PersistentClient` writes to `chroma_db/`. Unique IDs (`{source}::p{page}::c{index}`) prevent duplicates. Re-indexing deletes and recreates the collection.

### 10.5 Retrieval
Query is embedded with the same model. `collection.query(n_results=3)` is filtered by cosine distance ≤ `SIMILARITY_THRESHOLD`.

### 10.6 Generation
A strict system prompt instructs the model to answer only from context, ignore embedded instructions in documents, and return the exact refusal string when context is insufficient.

### 10.7 Refusal
Two triggers:
1. No chunk passes the threshold → refusal is returned without calling the LLM.
2. LLM returns the refusal string itself.

### 10.8 UI
Streamlit page with title, question input, Ask button, answer area, sources, expandable excerpts, and sidebar for re-indexing and status.

## 11. Testing and Evaluation

### 11.1 Unit tests
- `test_document_loader.py`, `test_chunker.py`, `test_retriever.py`, `test_rag_service.py`.
- 15 mocked tests run without Ollama and all passed successfully.

### 11.2 Evaluation dataset
50 questions:
- 20 direct
- 20 paraphrased
- 10 unanswerable

Each has an expected source (or null). Categories: attendance, credits, exams, placement, fees, handbook.

### 11.3 Final Evaluation Metrics

In our evaluation run, the system achieved the following results:

- **Retrieval Accuracy:** **97.50%** (39 out of 40 answerable questions retrieved the correct source document).
- **Safe Refusal Rate:** **100.00%** (10 out of 10 unanswerable questions were correctly refused).

*Human evaluation scores (1-5 scale for correctness, relevance, groundedness, clarity, and completeness) were also recorded in `evaluation_results.csv`.*

## 12. Limitations

- No OCR; scanned PDFs are skipped.
- Prompt alone cannot guarantee zero hallucinations.
- `llama3.2:3b` is small; answers should be verified.
- Similarity ≠ factual support.
- Threshold is a heuristic; needs tuning.
- English only.
- Single-user local deployment.

## 13. Conclusion

A complete, modular, locally hosted RAG assistant was built that answers student questions from college PDFs, cites sources, and refuses when evidence is insufficient. The project was managed in Jira across four sprints, tested with 15 unit tests and a 50-question evaluation set, and documented for viva and demonstration.

## 14. Future Scope

- OCR for scanned PDFs (`pytesseract` + `pdf2image`).
- Larger models (`llama3.1:8b`) for higher quality.
- Hybrid retrieval (BM25 + embeddings).
- Reranker (cross-encoder).
- Multi-user deployment with authentication.
- Table and figure extraction.
- Multi-language support.
- Feedback loop to improve threshold tuning.