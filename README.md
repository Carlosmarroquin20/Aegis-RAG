# Aegis-RAG

> **A production-grade RAG system where security is the architecture, not a checklist.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/mypy-strict-1f5082)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OWASP LLM](https://img.shields.io/badge/OWASP_LLM_Top_10-mitigated-d6262c)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

Aegis-RAG is a Retrieval-Augmented Generation API designed for environments where you cannot afford to ship "best effort" security. Every query is evaluated by a domain-level **Security Gateway** before any retrieval or generation happens, every LLM response is post-processed by an **Output Sanitizer**, and the whole pipeline is wrapped in a defence-in-depth middleware stack with first-class observability.

The codebase deliberately demonstrates senior-level practices that recruiters and tech leads care about: hexagonal architecture with swappable adapters, strict-typed Python, OWASP LLM Top 10 mitigations, Prometheus + Grafana out of the box, a hardened multi-stage Docker build, and a layered test suite (unit, integration, end-to-end).

---

## Engineering Highlights

- **Defence in depth, by design.** Five ASGI middlewares (RequestID, AccessLog, SecurityHeaders, APIKey, RateLimit) execute *before* any route handler, so 404s and 405s receive the same hardening as authenticated traffic.
- **Bounded label cardinality.** Prometheus path labels collapse to the matched FastAPI route template (`/api/v1/documents/{doc_id}`), never the raw URL — a subtle but production-critical decision.
- **Indirect prompt injection caught at the exit.** The `OutputSanitizer` blocks LLM responses that *echo* injection payloads from poisoned documents. The use case deliberately does not catch the error so it surfaces as a pipeline failure (HTTP 500), not a silent leak.
- **Magic-byte file detection.** Document uploads are dispatched to parsers based on actual file content, not the attacker-controllable `Content-Type` header.
- **Hexagonal for real.** The `LLMClientPort` and `VectorStorePort` interfaces mean swapping Ollama for OpenAI or ChromaDB for pgvector is a one-file change — and there are fakes in the test suite that prove it.
- **Air-gap compatible.** The full stack runs on local infrastructure (Ollama + ChromaDB). No data ever leaves the network.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Interface     FastAPI routes + middleware stack                     │
│                RequestID → AccessLog → SecurityHeaders →             │
│                RateLimit → APIKey → Routes                           │
├──────────────────────────────────────────────────────────────────────┤
│  Application   Use cases (QueryRAG, IngestDocuments) · DTOs          │
├──────────────────────────────────────────────────────────────────────┤
│  Domain        Models · Ports (abstract) · ChunkingService           │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure  ChromaDB · Ollama · Parsers · SecurityGateway       │
│                  OutputSanitizer · RateLimiter · Prometheus metrics  │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer depends only inward. Infrastructure adapters implement domain ports — no business logic is coupled to any vendor SDK.

### Request flow

```
HTTP Request
    │
    ▼  Middleware stack: correlation ID · access log · security headers · auth · rate limit
    │
    ▼  SecurityGateway.evaluate()      ← LLM01 · structural + Unicode NFC + 11 signatures + entropy
    │
    ▼  VectorStore.similarity_search() ← ChromaDB (swappable port)
    │
    ▼  LLMClient.generate()            ← Ollama (local, air-gapped)
    │
    ▼  OutputSanitizer.sanitize()      ← LLM02 · length cap · HTML strip · reflection guard · PII scan
    │
    ▼
JSON Response  (with X-Request-ID, X-Response-Time, X-RateLimit-* headers)
```

Document ingestion follows the same hexagonal pattern: magic-byte MIME detection → parser dispatch (TXT/MD/PDF/DOCX) → `ChunkingService` (paragraph → sentence → word fallback with overlap stitching) → content-addressed deduplication → vector store upsert.

---

## Security Controls — OWASP LLM Top 10

| Threat | Control | OWASP |
|---|---|---|
| Prompt injection | 11 regex signatures + Shannon entropy + Unicode NFC normalization | LLM01 |
| Insecure output handling | Length cap · HTML strip · reflection detection · PII flagging | LLM02 |
| System-prompt disclosure | Signature rules block probing queries (*"show me your prompt"*) | LLM07 |
| API abuse | Per-key sliding-window rate limiter with burst allowance | LLM06 |
| Sensitive data exposure | API key auth at ASGI level · non-root container · `Cache-Control: no-store` | LLM06 |
| File-upload attacks | Magic-byte MIME detection · 50 MB hard cap before parsing | General |
| Browser-side attacks | HSTS · CSP `default-src 'none'` · X-Frame-Options DENY · nosniff | General |

---

## Observability

`/metrics` exposes Prometheus-native metrics on the live port. The full observability stack (Prometheus + Grafana) is wired into `docker compose` and ships pre-provisioned — no manual import.

| Metric | Type | Purpose |
|---|---|---|
| `aegis_http_requests_total` | Counter | Request volume & error rate (SLO input) |
| `aegis_http_request_duration_seconds` | Histogram | Latency percentiles (p50/p95/p99) |
| `aegis_security_violations_total` | Counter | Blocked queries grouped by threat level |
| `aegis_security_rule_triggers_total` | Counter | Which injection signatures are firing |
| `aegis_output_reflections_total` | Counter | LLM responses blocked by reflection guard |
| `aegis_rag_queries_total` | Counter | Queries that reached the retrieval stage |

Logs are single-line JSON via `structlog` with an auto-bound `request_id` for end-to-end correlation between logs, metrics, traces, and the `X-Request-ID` response header.

The pre-provisioned Grafana dashboard surfaces request rate, error rate, p95 latency, and blocked queries as headline tiles, then breaks down HTTP and Security sections into full-resolution time series. Alert rules under `infra/prometheus/alerts.yml` fire on 5xx spikes, p95 regressions, security-violation floods, and any output-reflection event.

---

## Tech Stack

- **API & runtime** — FastAPI 0.115+, Uvicorn, Pydantic v2 (strict), `uv` package manager
- **RAG core** — ChromaDB · Ollama · `sentence-transformers/all-MiniLM-L6-v2` · pypdf · python-docx · markdown-it-py
- **Security & quality** — Ruff (lint + format) · Mypy strict · `pip-audit` · Trivy
- **Observability** — `structlog` · `prometheus-client` · Grafana with provisioned dashboards
- **Delivery** — Multi-stage Docker (non-root UID 1001) · GitHub Actions CI · docker-compose for local stack

---

## Quick Start

```bash
git clone https://github.com/your-username/aegis-rag.git
cd aegis-rag
cp .env.example .env
docker compose up -d
```

That single `docker compose up` starts the full stack — API, vector store, LLM, Prometheus, and a pre-loaded Grafana dashboard:

| Service | URL | Notes |
|---|---|---|
| Aegis-RAG API | `http://localhost:8000` | Main application |
| ChromaDB | `http://localhost:8001` | Vector store |
| Ollama | `http://localhost:11434` | Pulls `llama3.2` on first start |
| Prometheus | `http://localhost:9090` | Scrape + alerting |
| **Grafana** | **`http://localhost:3000`** | **Live dashboard, anonymous viewer access** |

Then index a document and run a query:

```bash
# Upload
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-API-Key: dev-key-change-in-production" \
  -F "file=@./README.md"

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "What is the security model?", "top_k": 5}'
```

Or run the same flow as a single test: `uv run pytest -m e2e --no-cov`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe (no dependencies) |
| `GET`  | `/ready`  | Readiness probe (checks ChromaDB + Ollama) |
| `GET`  | `/metrics` | Prometheus scrape target |
| `POST` | `/api/v1/query` | Submit a question to the RAG pipeline |
| `POST` | `/api/v1/documents` | Upload and index a document (TXT, MD, PDF, DOCX) |
| `GET`  | `/api/v1/documents` | List indexed chunks (paginated) |
| `DELETE` | `/api/v1/documents/{id}` | Delete a single chunk |
| `DELETE` | `/api/v1/documents` | Bulk delete by ID list (max 500) |

Interactive Swagger docs are served at `http://localhost:8000/docs` when `DEBUG=true`.

---

## Testing

The suite is layered so each level catches what the others cannot.

```bash
uv run pytest                       # unit + integration, with coverage gate
uv run pytest tests/unit/           # fast, no I/O, ~80 tests
uv run pytest tests/integration/    # in-memory adapters, ingestion pipeline
uv run pytest -m e2e --no-cov       # full HTTP surface against docker-compose
```

**End-to-end tests** issue real requests through the live middleware stack. They auto-skip with a clear message if `docker compose` is not up, so CI never fails on a forgotten container. Point them at any environment via env vars:

```bash
AEGIS_E2E_BASE_URL=https://staging.example.com \
AEGIS_E2E_API_KEY=<key> uv run pytest -m e2e --no-cov
```

**Lint, format, type-check:**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

**CI pipeline** (GitHub Actions, every push):

```
Ruff lint → Ruff format → Mypy strict → pip-audit → Unit tests + coverage → Docker build → Trivy scan
```

---

## Project Structure

```
src/aegis/
├── domain/           # Models · Ports · ChunkingService            (no I/O)
├── application/      # Use cases · DTOs                            (orchestration)
├── infrastructure/   # ChromaDB · Ollama · Parsers · Security      (adapters)
│   └── observability/  # Prometheus metrics
└── interface/api/    # FastAPI app · middleware · routes           (HTTP boundary)

infra/
├── prometheus/       # Scrape config + alert rules
├── grafana/          # Dashboard JSON + provisioning
└── terraform/        # Cloud infrastructure (optional)

tests/
├── unit/             # Pure-function tests + middleware + use case
├── integration/      # Use case ↔ adapter wiring with in-memory fakes
└── e2e/              # Full stack roundtrip (opt-in via -m e2e)
```

---

## License

MIT — see [LICENSE](LICENSE).
