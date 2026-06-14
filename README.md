# NeuroDocs AI

> **Production-grade Retrieval-Augmented Generation** combining dense vector search, sparse keyword retrieval, reciprocal rank fusion, cross-encoder reranking, and a locally-hosted LLM — all served through a FastAPI backend with a modern web UI.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Screenshots](#screenshots)
5. [Project Structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the System](#running-the-system)
10. [API Reference](#api-reference)
11. [How It Works](#how-it-works)
12. [Evaluation](#evaluation)
13. [Troubleshooting](#troubleshooting)
14. [Roadmap](#roadmap)
15. [License](#license)

---

## Overview

This system ingests PDF documents, builds a hybrid search index, and answers natural-language questions using a fully local LLM (via [Ollama](https://ollama.com)). No data ever leaves your machine.

| Component | Technology |
|-----------|-----------|
| Embeddings | BGE-M3 (`BAAI/bge-m3`) |
| Vector Store | FAISS (cosine similarity) |
| Keyword Search | BM25 (TF-IDF sparse retrieval) |
| Fusion Strategy | Reciprocal Rank Fusion (RRF) |
| Reranker | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM | Ollama (local — any model: `llama3`, `mistral`, etc.) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS / HTML5 / CSS3 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│                                                          │
│  PDF Upload → PyMuPDF → Chunker → BGE-M3 Embeddings     │
│                                  ↓            ↓          │
│                               FAISS         BM25         │
│                            (dense idx)  (sparse idx)     │
└─────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│                                                          │
│  User Query                                              │
│      ↓                                                   │
│  BGE-M3 Query Embedding                                  │
│      ↓                    ↓                              │
│  FAISS Search          BM25 Search                       │
│      ↓                    ↓                              │
│       └──── RRF Fusion ───┘                              │
│                   ↓                                      │
│          Cross-Encoder Reranker                          │
│                   ↓                                      │
│          Top-K Context Chunks                            │
│                   ↓                                      │
│            Ollama LLM (local)                            │
│                   ↓                                      │
│              Final Answer                                │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Hybrid Retrieval** — Combines semantic (FAISS dense) and lexical (BM25 sparse) search for superior recall across both conceptual and keyword-specific queries.
- **Reciprocal Rank Fusion** — Merges ranked lists from multiple retrievers without requiring score normalization.
- **Cross-Encoder Reranking** — Applies a fine-grained relevance model on the fused candidate set to surface the most pertinent chunks.
- **BGE-M3 Embeddings** — State-of-the-art multilingual, multi-granularity embeddings supporting 100+ languages.
- **Local LLM via Ollama** — Zero data egress; swap any Ollama-compatible model without code changes.
- **FastAPI Backend** — Async, high-performance REST API with automatic OpenAPI docs at `/docs`.
- **PDF Upload UI** — Drag-and-drop PDF ingestion directly from the browser.
- **Streaming Responses** — Token-by-token answer streaming via WebSockets.
- **Modular Pipeline** — Each component (chunker, embedder, retriever, reranker, LLM) is independently swappable.
- **Evaluation Suite** — Built-in scripts for measuring retrieval precision, recall, and answer quality.

---

## Screenshots

> Place your screenshots in `docs/screenshots/` and they will render automatically on GitHub.

### Homepage
![Homepage](docs/screenshots/homepage.png)

### Chat Interface
![Chat Interface](docs/screenshots/chat_interface.png)

### Upload Interface
![Upload Interface](docs/screenshots/upload_interface.png)

---

## Project Structure

```
advanced-rag-system/
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   └── upload.py               # PDF upload & ingestion endpoint
│
├── data/                       # Raw source documents (PDFs)
├── docs/                       # Project documentation
│   └── screenshots/            # UI screenshots (homepage, chat, upload)
│       ├── homepage.png
│       ├── chat_interface.png
│       └── upload_interface.png
├── evaluation/                 # Retrieval & answer quality evaluation scripts
│
├── frontend/                   # Browser-based UI
│   ├── app.js                  # Chat logic, WebSocket client, upload handler
│   ├── index.html              # Single-page application shell
│   └── style.css               # Responsive stylesheet
│
├── indexes/                    # Persisted FAISS index & BM25 artifacts
│
├── ingestion/                  # Document processing pipeline
│   ├── chunker.py              # Recursive/semantic text chunking
│   ├── embedder.py             # BGE-M3 embedding wrapper
│   ├── embedding.py            # Batch embedding utilities
│   ├── pdf_loader.py           # PyMuPDF-based PDF text extractor
│   └── rebuild_pipeline.py     # CLI: re-index all documents from scratch
│
├── llm/                        # LLM integration
│   └── ...                     # Ollama client, prompt templates, streaming
│
├── memory/                     # Conversation memory / context management
│
├── retrieval/                  # Search & fusion logic
│   └── ...                     # FAISS search, BM25 search, RRF, reranker
│
├── tests/                      # Unit and integration tests
├── uploads/                    # Temporary upload staging directory
├── venv/                       # Python virtual environment (git-ignored)
│
├── main.py                     # Application entry point
├── rag_chat.py                 # Core RAG orchestration logic
├── requirements.txt            # Python dependencies (pinned)
└── README.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | 3.11+ recommended |
| Ollama | Latest | [Install guide](https://ollama.com/download) |
| RAM | ≥ 16 GB | 32 GB recommended for large corpora |
| Disk | ≥ 10 GB | For models + indexes |
| GPU | Optional | CUDA 11.8+ for GPU-accelerated FAISS |

### Install & start Ollama

**macOS**
```bash
brew install ollama
# or download the .dmg from https://ollama.com/download/mac
ollama serve          # starts the Ollama daemon
```

**Windows**
```powershell
# Download and run the installer from https://ollama.com/download/windows
# After installation, Ollama runs automatically as a background service.
# To start it manually from PowerShell:
ollama serve
```

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve          # starts the Ollama daemon (or runs via systemd automatically)
```

**Pull a model and verify**
```bash
# Choose any model — llama3 is a good default
ollama pull llama3
ollama pull mistral   # alternative: smaller, faster
ollama pull phi3      # alternative: very lightweight

# Confirm the daemon is running and models are available
ollama list
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/advanced-rag-system.git
cd advanced-rag-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) GPU-accelerated FAISS
# pip uninstall faiss-cpu && pip install faiss-gpu
```

> **First run:** BGE-M3 and the cross-encoder model (~1.5 GB combined) will be downloaded automatically from HuggingFace Hub on first launch.

---

## Configuration

Create a `.env` file in the project root (or edit `main.py` directly):

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Retrieval
FAISS_TOP_K=20
BM25_TOP_K=20
RRF_TOP_K=10
RERANKER_TOP_K=5

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EMBEDDING_DEVICE=cpu           # or "cuda" / "mps"

# Paths
INDEX_DIR=./indexes
UPLOAD_DIR=./uploads
```

---

## Running the System

### Start the API server

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Open the UI

```
http://127.0.0.1:8000/ui
```

### Re-index all documents

```bash
python ingestion/rebuild_pipeline.py --data-dir ./data
```

---

## API Reference

Interactive docs are auto-generated by FastAPI:

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and ingest a PDF |
| `POST` | `/api/chat` | Send a query, receive an answer |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document from the index |
| `GET` | `/health` | Health check |

**Example — Upload a PDF:**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

**Example — Query:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings in chapter 3?"}'
```

---

## How It Works

### 1. Ingestion

1. `pdf_loader.py` extracts raw text from each page using PyMuPDF.
2. `chunker.py` splits text into overlapping chunks (default 512 tokens, 64-token overlap) using LangChain's `RecursiveCharacterTextSplitter`.
3. `embedder.py` encodes every chunk with **BGE-M3**, producing 1024-dimensional dense vectors.
4. Vectors are stored in a **FAISS** flat index; raw text is stored alongside for BM25.

### 2. Retrieval

For each user query:

1. The query is embedded with BGE-M3.
2. **FAISS** returns the top-K nearest-neighbor chunks (dense retrieval).
3. **BM25** returns the top-K term-frequency-matched chunks (sparse retrieval).
4. **RRF** merges both ranked lists: score = Σ 1 / (rank + k), where k=60.
5. The **cross-encoder reranker** scores each candidate against the query and returns the final top-K.

### 3. Generation

The reranked context chunks are injected into a prompt template and sent to the **Ollama LLM**. The answer streams back token-by-token to the frontend via WebSocket.

---

## Evaluation

```bash
# Run the evaluation suite against a ground-truth QA dataset
python evaluation/run_eval.py --dataset evaluation/qa_pairs.json

# Metrics reported:
#   Retrieval: Recall@K, Precision@K, MRR, NDCG
#   Generation: ROUGE-L, BERTScore, Faithfulness
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Connection refused` on port 11434 | Ollama not running | Run `ollama serve` |
| CUDA out of memory | GPU VRAM too low | Set `EMBEDDING_DEVICE=cpu` |
| Empty answers / hallucinations | Context chunks too short | Increase `CHUNK_SIZE` |
| Slow first query | Model cold start | Pre-warm: send a dummy query after startup |
| `ModuleNotFoundError: faiss` | Package not installed | `pip install faiss-cpu` |
| BGE-M3 download fails | HuggingFace rate limit | Set `HF_TOKEN` in `.env` |

---

## Roadmap

- [ ] Multi-document conversation memory with session management
- [ ] Metadata filtering (date range, document source, author)
- [ ] Hybrid chunking: semantic + recursive fallback
- [ ] Docker Compose deployment (app + Ollama in containers)
- [ ] OpenAI / Anthropic API fallback when Ollama is unavailable
- [ ] Incremental index updates (add/remove docs without full rebuild)
- [ ] REST streaming via SSE in addition to WebSocket

---

## License

```
MIT License

Copyright (c) 2026 Abdul Kareem K

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with BGE-M3 · FAISS · BM25 · RRF · Cross-Encoder · Ollama · FastAPI
</p>