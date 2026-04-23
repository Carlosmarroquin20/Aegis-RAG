# Aegis-RAG

**Security-hardened Retrieval-Augmented Generation API for enterprise environments.**

Aegis-RAG routes every query through an OWASP LLM Top 10 security pipeline before it reaches the RAG engine. The result is a production-grade system where prompt injection, output manipulation, and data exfiltration are mitigated by default rather than bolted on as an afterthought.

> Built with hexagonal architecture, structured observability, and a full CI/CD pipeline — designed to demonstrate real-world MLOps, AppSec, and backend engineering practices.

---

## Key Features

- **Prompt injection defence** — 11 heuristic signatures + Shannon entropy analysis, applied after NFC Unicode normalization to defeat homoglyph and encoding attacks.
- **Output sanitization** — length enforcement, HTML stripping, prompt-reflection detection, and PII pattern flagging on every LLM response.
- **Defence-in-depth middleware** — API key authentication, sliding-window rate limiting, security response headers (HSTS, CSP, X-Frame-Options), request ID correlation, and structured access logging with response timing.
- **Adapter-based architecture** — swap ChromaDB for pgvector, or Ollama for OpenAI, by changing a single file. No domain code is affected.
- **Air-gap compatible** — runs entirely on local infrastructure (Ollama + ChromaDB). No data leaves the network.
- **Prometheus-ready** — `/metrics` endpoint exposes request counters, latency histograms, and security violations for SRE dashboards and alerting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Interface          FastAPI routes, ASGI middleware stack           │
│                     RequestID → AccessLog → SecurityHeaders →      │
│                     RateLimit → APIKey → Routes                    │
├─────────────────────────────────────────────────────────────────────┤
│  Application        Use cases (QueryRAG, IngestDocuments), DTOs    │
├─────────────────────────────────────────────────────────────────────┤
│  Domain             Models, Ports (abstract interfaces),           │
│                     ChunkingService                                │
├─────────────────────────────────────────────────────────────────────┤
│  Infrastructure     ChromaDB adapter, Ollama adapter, Parsers,     │
│                     SecurityGateway, OutputSanitizer, RateLimiter  │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer depends only inward. Infrastructure adapters implement domain ports — no business logic is coupled to any vendor SDK.

---

## Security Controls (OWASP LLM Top 10)

| Threat | Control | OWASP |
|---|---|---|
| Prompt injection | 11 regex signatures + entropy scoring + Unicode NFC normalization | LLM01 |
| Insecure output handling | Length cap, HTML stripping, reflection detection, PII flagging | LLM02 |
| System prompt disclosure | Signature rules block probing queries ("show me your prompt") | LLM07 |
| API abuse | Per-key sliding-window rate limiter with burst allowance | LLM06 |
| Sensitive data exposure | API key auth at ASGI level, non-root container, no-store cache | LLM06 |
| File upload attacks | Magic-byte MIME detection, 50 MB hard cap before parsing | General |
| Browser-side attacks | HSTS, CSP, X-Frame-Options, X-Content-Type-Options on every response | General |

### Query Pipeline

```
HTTP Request
    │
    ▼
RequestIDMiddleware         ← assigns correlation ID (or reuses gateway-provided)
    │
    ▼
AccessLogMiddleware         ← structured log: method, path, status, duration_ms, IP
    │
    ▼
SecurityHeadersMiddleware   ← HSTS, CSP, X-Frame-Options, nosniff, no-store
    │
    ▼
APIKeyMiddleware            ← identity enforcement (ASGI layer)
    │
    ▼
RateLimitMiddleware         ← per-key sliding-window quota + X-RateLimit-* headers
    │
    ▼
SecurityGateway.evaluate()
  ├─ Structural validation (length, null bytes)
  ├─ Unicode NFC normalization
  ├─ Heuristic threat scoring (injection signatures + entropy)
  └─ Deep sanitization (HTML/template stripping)
    │
    ▼
VectorStore.similarity_search()   ← ChromaDB (adapter pattern)
    │
    ▼
LLMClient.generate()              ← Ollama (local, air-gapped)
    │
    ▼
OutputSanitizer.sanitize()
  ├─ Length truncation
  ├─ HTML stripping
  ├─ Reflection detection          ← blocks if LLM echoed injection payload
  └─ PII pattern flagging
    │
    ▼
JSON Response (with X-Request-ID + X-Response-Time)
```

### Ingestion Pipeline

```
File Upload (multipart/form-data)
    │
    ▼
File size check (< 50 MB)
    │
    ▼
ParserRegistry.detect_mime_type()  ← magic bytes, not Content-Type
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

## Observability

Aegis-RAG ships with **Prometheus-native metrics** exposed at `/metrics` for scraping.

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `aegis_http_requests_total` | Counter | `method`, `path`, `status` | Request volume & error rate (SLO input) |
| `aegis_http_request_duration_seconds` | Histogram | `method`, `path` | Latency percentiles (p50, p95, p99) |
| `aegis_security_violations_total` | Counter | `threat_level` | Blocked queries grouped by severity |
| `aegis_security_rule_triggers_total` | Counter | `rule` | Which injection signatures are hitting |
| `aegis_output_reflections_total` | Counter | — | LLM outputs blocked by reflection detection |
| `aegis_rag_queries_total` | Counter | — | Queries that reached the retrieval stage |

Path labels use the matched FastAPI route template (e.g. `/api/v1/documents/{doc_id}`) — not the raw URL — to keep label cardinality bounded.

Every log line is a single JSON object (via `structlog`) with an auto-bound `request_id` field for end-to-end correlation between logs, metrics, and the `X-Request-ID` response header.

Example scrape config:

```yaml
scrape_configs:
  - job_name: aegis-rag
    metrics_path: /metrics
    static_configs:
      - targets: ["aegis-rag:8000"]
```

---

## Tech Stack

| Concern | Technology |
|---|---|
| API framework | FastAPI 0.115+, Uvicorn |
| Validation | Pydantic v2, pydantic-settings |
| Vector store | ChromaDB (adapter pattern — swappable for pgvector, Pinecone, etc.) |
| LLM backend | Ollama (local inference, air-gap compatible) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Document parsing | pypdf, python-docx, markdown-it-py, filetype (magic-byte MIME) |
| Observability | structlog (JSON in prod), request ID correlation, Prometheus `/metrics` |
| Linting | Ruff (lint + format), Mypy (strict mode) |
| Security scanning | pip-audit (dependencies), Trivy (container image) |
| Package manager | uv |
| Containerization | Docker multi-stage build, non-root runtime (UID 1001) |
| CI/CD | GitHub Actions — lint → security scan → test → Docker build + Trivy |

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [Ollama](https://ollama.ai/) (optional — included in compose stack)

### 1. Clone and install

```bash
git clone https://github.com/your-username/aegis-rag.git
cd aegis-rag
uv sync --dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set VALID_API_KEYS and adjust model/store settings as needed
```

### 3. Start the stack

```bash
docker compose up -d
```

This starts:

| Service | URL | Purpose |
|---|---|---|
| Aegis-RAG API | `http://localhost:8000` | Main application |
| ChromaDB | `http://localhost:8001` | Vector store |
| Ollama | `http://localhost:11434` | Local LLM inference |

### 4. Index a document

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-API-Key: dev-key-change-in-production" \
  -F "file=@./path/to/document.pdf"
```

### 5. Query the RAG pipeline

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
| `GET` | `/health` | Liveness probe (no dependencies) |
| `GET` | `/ready` | Readiness probe (checks ChromaDB + Ollama) |
| `GET` | `/metrics` | Prometheus metrics (scrape target) |
| `POST` | `/api/v1/query` | Submit a question to the RAG pipeline |
| `POST` | `/api/v1/documents` | Upload and index a document (TXT, MD, PDF, DOCX) |
| `GET` | `/api/v1/documents` | List indexed document chunks (paginated) |
| `DELETE` | `/api/v1/documents/{id}` | Delete a single chunk |
| `DELETE` | `/api/v1/documents` | Bulk delete by ID list (max 500) |

Interactive docs available at `http://localhost:8000/docs` when `DEBUG=true`.

---

## Development

### Run tests

```bash
uv run pytest                        # all tests with coverage
uv run pytest tests/unit/            # unit tests only
uv run pytest tests/integration/     # integration tests (requires running stack)
```

### Lint and type-check

```bash
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # auto-format
uv run mypy src/                     # strict type checking
```

### CI Pipeline

Every push triggers the full pipeline in GitHub Actions:

```
Ruff lint → Ruff format → Mypy → pip-audit → Unit tests + coverage → Docker build → Trivy scan
```

---

## Project Structure

```
src/aegis/
├── config.py                          # pydantic-settings: all env vars, validation
├── domain/
│   ├── models/                        # Pure data models (no I/O)
│   ├── ports/                         # Abstract interfaces (VectorStore, LLM, Parser)
│   └── services/                      # ChunkingService
├── application/
│   ├── dtos/                          # Request/response contracts
│   └── use_cases/                     # QueryRAG, IngestDocuments
├── infrastructure/
│   ├── llm/                           # OllamaAdapter (implements LLMClientPort)
│   ├── observability/                 # Prometheus metrics (counters, histograms)
│   ├── parsers/                       # TXT, Markdown, PDF, DOCX + ParserRegistry
│   ├── security/                      # SecurityGateway, OutputSanitizer, RateLimiter
│   └── vector_stores/                 # ChromaDBAdapter (implements VectorStorePort)
└── interface/
    └── api/
        ├── main.py                    # FastAPI app factory + lifespan
        ├── dependencies.py            # Dependency injection wiring
        ├── middleware/                 # RequestID, AccessLog, SecurityHeaders,
        │                              # APIKey, RateLimit
        └── routes/                    # query, documents, health
```

---

## License

MIT
