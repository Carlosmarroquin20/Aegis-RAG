# Aegis-RAG

A hardened Retrieval-Augmented Generation (RAG) system designed for corporate environments.
Built with a security-first architecture that enforces OWASP LLM Top 10 controls at every layer of the pipeline.

> **Portfolio project** — demonstrates production-grade MLOps/DevOps practices including clean architecture, AppSec integration, observability, and containerized deployment.

---

## Architecture

Aegis-RAG follows a **hexagonal (ports & adapters)** architecture with four strict layers:

```
┌─────────────────────────────────────────────────────┐
│  Interface Layer     FastAPI routes, ASGI middleware │
├─────────────────────────────────────────────────────┤
│  Application Layer   Use cases, DTOs                │
├─────────────────────────────────────────────────────┤
│  Domain Layer        Models, ports (interfaces)     │
│                      ChunkingService                │
├─────────────────────────────────────────────────────┤
│  Infrastructure      ChromaDB, Ollama, Parsers,     │
│                      SecurityGateway, RateLimiter   │
└─────────────────────────────────────────────────────┘
```

Each layer depends only inward. Swapping ChromaDB for pgvector, or Ollama for OpenAI,
requires a change in a single adapter file.

---

## Security Design (OWASP LLM Top 10)

| Control | Implementation | Coverage |
|---|---|---|
| Prompt injection detection | 11 heuristic signatures + Shannon entropy analysis | LLM01 |
| Unicode normalization | NFC collapse before pattern matching (defeats homoglyph attacks) | LLM01 |
| Instruction isolation in LLM | Hardcoded system prompt refusing document-embedded directives | LLM01/LLM02 |
| Output sanitization | Length cap, HTML stripping, reflection detection, PII flagging | LLM02 |
| API key authentication | ASGI-level middleware (runs before 404 responses) | LLM06 |
| Per-key rate limiting | Sliding-window algorithm, Redis-swappable backend | LLM06 |
| System prompt disclosure | Signature rules block probing queries | LLM07 |
| File type validation | Magic-byte MIME detection (ignores attacker-controlled Content-Type) | General |
| File size enforcement | 50 MB cap before parsing (prevents decompression bombs) | General |
| Non-root container | UID 1001 in Docker runtime stage | General |

### Query Pipeline

```
HTTP Request
    │
    ▼
APIKeyMiddleware          ← identity enforcement (ASGI layer)
    │
    ▼
RateLimitMiddleware       ← per-key sliding-window quota
    │
    ▼
SecurityGateway.evaluate()
  ├─ Structural validation (length, null bytes)
  ├─ Unicode normalization
  ├─ Heuristic threat scoring (injection signatures)
  └─ Deep sanitization (HTML/template stripping)
    │
    ▼
VectorStore.similarity_search()   ← ChromaDB
    │
    ▼
LLMClient.generate()              ← Ollama (local, private)
    │
    ▼
OutputSanitizer.sanitize()
  ├─ Length truncation
  ├─ HTML stripping
  ├─ Reflection detection         ← blocks if LLM echoed injection payload
  └─ PII pattern flagging
    │
    ▼
JSON Response
```

### Ingestion Pipeline

```
File Upload (multipart/form-data)
    │
    ▼
File size check (< 50 MB)
    │
    ▼
ParserRegistry.detect_mime_type() ← magic bytes, not Content-Type
    │
    ▼
Parser (TXT / Markdown / PDF / DOCX)
    │
    ▼
ChunkingService
  ├─ Paragraph splitting
  ├─ Sentence-boundary splitting
  ├─ Word-boundary fallback
  └─ Overlap stitching + content-addressed IDs
    │
    ▼
Deduplication (hash-based, idempotent upsert)
    │
    ▼
ChromaDB upsert
```

---

## Tech Stack

| Concern | Technology |
|---|---|
| API | FastAPI 0.115+, uvicorn |
| Validation | Pydantic v2, pydantic-settings |
| Vector store | ChromaDB (adapter pattern — swappable) |
| LLM | Ollama (local inference, air-gap compatible) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Document parsing | pypdf, python-docx, markdown-it-py |
| Observability | structlog (JSON in prod, console in dev) |
| Package manager | uv |
| Containerization | Docker multi-stage + docker-compose |
| CI | GitHub Actions (lint → scan → test → Docker build) |

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [Ollama](https://ollama.ai/) (optional for local dev without containers)

### 1. Clone and install

```bash
git clone https://github.com/your-org/aegis-rag.git
cd aegis-rag
uv sync --dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set VALID_API_KEYS and adjust model/store settings
```

### 3. Start the stack

```bash
docker compose up -d
```

This starts:
- **Aegis-RAG API** on `http://localhost:8000`
- **ChromaDB** on `http://localhost:8001`
- **Ollama** on `http://localhost:11434` (pulls `llama3.2` on first start)

### 4. Index documents

```bash
uv run python scripts/seed_documents.py --source-dir ./data/docs
```

### 5. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "What is the remote work policy?", "top_k": 5}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks ChromaDB + Ollama) |
| `POST` | `/api/v1/query` | Submit a question to the RAG pipeline |
| `POST` | `/api/v1/documents` | Upload and index a document (TXT, MD, PDF, DOCX) |
| `GET` | `/api/v1/documents` | List indexed document chunks |
| `DELETE` | `/api/v1/documents/{id}` | Delete a chunk by ID |
| `DELETE` | `/api/v1/documents` | Bulk delete by ID list |

Interactive docs available at `http://localhost:8000/docs` when `DEBUG=true`.

---

## Development

### Run tests

```bash
uv run pytest                        # all tests with coverage
uv run pytest tests/unit/            # unit tests only
uv run pytest tests/integration/     # integration tests
```

### Lint and type-check

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

### Dev Container

Open in VS Code with the Remote - Containers extension. The `.devcontainer/` configuration sets up the full stack automatically, including pre-commit hooks.

---

## Project Structure

```
src/aegis/
├── config.py                          # pydantic-settings, all env vars
├── domain/
│   ├── models/                        # Pure data models (no I/O)
│   ├── ports/                         # Abstract interfaces (VectorStore, LLM, Parser)
│   └── services/                      # ChunkingService
├── application/
│   ├── dtos/                          # Request/response contracts
│   └── use_cases/                     # QueryRAG, IngestDocuments
├── infrastructure/
│   ├── llm/                           # OllamaAdapter
│   ├── parsers/                       # TXT, Markdown, PDF, DOCX + ParserRegistry
│   ├── security/                      # SecurityGateway, RateLimiter, OutputSanitizer
│   └── vector_stores/                 # ChromaDBAdapter
└── interface/
    └── api/
        ├── main.py                    # FastAPI app factory + lifespan
        ├── dependencies.py            # Object graph wiring
        ├── middleware/                # APIKeyMiddleware, RateLimitMiddleware
        └── routes/                    # query, documents, health
```

---

## License

MIT
