# RAG-Based Knowledge Retrieval API

A **Retrieval-Augmented Generation (RAG) backend service** built using **FastAPI**, **ChromaDB**, and **Groq LLM API**.  
The system enables semantic question answering over custom knowledge by combining vector search with large language models.

This project is designed as a **backend-first, API-driven RAG system**, suitable for demos, experimentation, and learning modern AI-backed backend architectures.

---

## Tech Stack

- **FastAPI** – API framework
- **ChromaDB** – Embedded vector database for semantic retrieval
- **Groq API** – External LLM inference (`llama-3.1-8b-instant`)
- **Python**

---

## How It Works (High Level)

1. Knowledge is stored as vector embeddings in **ChromaDB**
2. User queries are semantically matched against stored knowledge
3. Retrieved context is passed to an LLM (via Groq)
4. The LLM generates a context-aware answer

---


## Live Demo

The API is deployed and accessible via **interactive Swagger UI**:

**Live Demo:** [`LINK`](https://rag-based-knowledge-retrieval-api.onrender.com/docs)

You can:
- Add new knowledge
- Ask questions
- See answers immediately

No setup required.

---

## API Endpoints

### `GET /health`
Health check endpoint used for uptime monitoring.

**Response**
```json
{
  "status": "ok",
  "service": "rag-api",
  "vector_store": "chromadb"
}
