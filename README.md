# Aegis-RAG

> **A production-grade RAG system where security is the architecture, not a checklist.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/mypy-strict-1f5082)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OWASP LLM](https://img.shields.io/badge/OWASP_LLM_Top_10-mitigated-d6262c)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[![CI](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-~85%25-brightgreen)](#testing)

Aegis-RAG is a Retrieval-Augmented Generation API designed for environments where you cannot afford to ship "best effort" security. Every query is evaluated by a domain-level **Security Gateway** before any retrieval or generation happens, every LLM response is post-processed by an **Output Sanitizer**, and the whole pipeline is wrapped in a defence-in-depth middleware stack with first-class observability.

The codebase deliberately demonstrates senior-level practices that recruiters and tech leads care about: hexagonal architecture with swappable adapters, strict-typed Python, OWASP LLM Top 10 mitigations, logs + metrics + distributed traces out of the box, a hardened multi-stage Docker build, a layered test suite (unit, integration, end-to-end), AWS Terraform, and a React demo console.

<p align="center">
  <img src="docs/security-demo.gif" alt="Aegis-RAG web console — prompt-injection probes blocked by the live gateway" width="720">
  <br>
  <em>The web console's security demo: fire prompt-injection payloads at the live gateway and watch them get blocked before any retrieval happens.</em>
</p>

---

## Engineering Highlights

- **Defence in depth, by design.** The ASGI middleware chain (RequestID → AccessLog → SecurityHeaders → CORS → APIKey → RateLimit) executes *before* any route handler, so 404s and 405s receive the same hardening as authenticated traffic. RequestID sits outermost, so every log line and span carries the same correlation ID.
- **Bounded label cardinality.** Prometheus path labels collapse to the matched FastAPI route template (`/api/v1/documents/{doc_id}`), never the raw URL — a subtle but production-critical decision.
- **Indirect prompt injection caught at the exit.** The `OutputSanitizer` blocks LLM responses that *echo* injection payloads from poisoned documents. The use case deliberately does not catch the error so it surfaces as a pipeline failure (HTTP 500), not a silent leak.
- **Magic-byte file detection.** Document uploads are dispatched to parsers based on actual file content, not the attacker-controllable `Content-Type` header.
- **Hexagonal for real.** The `LLMClientPort`, `VectorStorePort` and `RateLimitStore` interfaces mean swapping Ollama for OpenAI, ChromaDB for pgvector, or the in-memory rate limiter for the shared **Redis** backend is a one-file change — and there are fakes in the test suite that prove it.
- **Three pillars of observability.** Structured JSON logs, Prometheus metrics, and opt-in **OpenTelemetry** traces (a span per pipeline stage) all correlate through a single `request_id` / `trace_id`.
- **Air-gap compatible.** The full stack runs on local infrastructure (Ollama + ChromaDB). No data ever leaves the network.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Interface     FastAPI routes + middleware stack                     │
│                RequestID → AccessLog → SecurityHeaders → CORS →      │
│                APIKey → RateLimit → Routes                           │
├──────────────────────────────────────────────────────────────────────┤
│  Application   Use cases (QueryRAG, IngestDocuments) · DTOs          │
├──────────────────────────────────────────────────────────────────────┤
│  Domain        Models · Ports (abstract) · ChunkingService           │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure  ChromaDB · Ollama · Parsers · SecurityGateway       │
│                  OutputSanitizer · RateLimiter (memory/Redis)        │
│                  Prometheus metrics · OpenTelemetry tracing          │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer depends only inward. Infrastructure adapters implement domain ports — no business logic is coupled to any vendor SDK.

The runtime topology — everything a single `docker compose up` starts, its trust boundaries, and the observability fan-out:

```mermaid
flowchart LR
    client(["Web Console<br/>React SPA · :8080"])
    subgraph stack["Local stack · docker compose (aegis-net) — air-gapped by design"]
      direction LR
      subgraph ingress["Ingress · API key · rate limit · security headers"]
        api["Aegis-RAG API<br/>FastAPI · hexagonal :8000"]
      end
      redis[("Redis<br/>rate-limit store")]
      chroma[("ChromaDB<br/>vector store")]
      ollama["Ollama<br/>LLM · llama3.2"]
      otel["OTel Collector<br/>OTLP · spanmetrics"]
      jaeger["Jaeger<br/>traces"]
      prom["Prometheus<br/>metrics"]
      grafana["Grafana<br/>dashboards"]
    end
    client -->|query · ingest| api
    api --> chroma
    api --> ollama
    api -.->|rate limit| redis
    api -.->|OTLP spans| otel
    otel -.->|traces| jaeger
    prom -.->|scrape| otel
    grafana -->|query| prom
    style api stroke:#e11d63,stroke-width:2px
```

> **▶ [Open the interactive system map](https://carlosmarroquin20.github.io/Aegis-RAG/aegis-architecture.html)** — the same topology as an explorable map (pan/zoom, guided views, theme toggle), authored as a validated [JSON spec](docs/diagrams/aegis-architecture.json) (diagram-as-code) and rendered with [Archify](https://github.com/tt-a1i/archify).

### Request flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as Aegis-RAG API
    participant G as SecurityGateway
    participant S as ChromaDB
    participant L as Ollama
    participant O as OutputSanitizer
    C->>A: POST /api/v1/query
    Note over A: middleware — API key · rate limit · security headers
    A->>G: evaluate(query) · LLM01
    G-->>A: CLEAN
    A->>S: similarity_search
    S-->>A: top-k chunks
    A->>L: generate(context, query)
    L-->>A: answer
    A->>O: sanitize(answer) · LLM02
    O-->>A: clean
    A-->>C: 200 JSON {answer, sources}
    Note over A,G: a BLOCKED verdict short-circuits to HTTP 400 — no retrieval, no LLM call
```

> **▶ [Open the interactive sequence](https://carlosmarroquin20.github.io/Aegis-RAG/aegis-query-sequence.html)** — the same security-first flow as an explorable diagram with activation bars and phase bands, as a [JSON spec](docs/diagrams/aegis-query-sequence.json) (diagram-as-code, rendered with Archify).

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

**Distributed tracing (OpenTelemetry)** is opt-in (`TRACING_ENABLED`). When on, the FastAPI server span and the HTTPX call to Ollama are auto-instrumented, the query use case emits a child span per pipeline stage (`security.evaluate` → `rag.retrieve` → `llm.generate` → `output.sanitize`), and every log line carries the active `trace_id`/`span_id` so logs pivot to their trace. Spans export over OTLP/HTTP to an **OpenTelemetry Collector**, which fans them out two ways: the raw traces to a Jaeger backend at `http://localhost:16686`, and — via the **spanmetrics connector** — trace-derived RED metrics to Prometheus. The SDK lives in the optional `otel` extra, so the base image stays lean when tracing is off.

Two pre-provisioned Grafana dashboards ship with the stack. **Production Overview** surfaces request rate, error rate, p95 latency, and blocked queries as headline tiles, then breaks down HTTP and Security into full-resolution time series. **Trace Analytics** turns the pipeline spans into per-stage RED charts (call rate, errors, and p50/p95/p99 latency for `security.evaluate` / `rag.retrieve` / `llm.generate` / `output.sanitize`), plus a SecurityGateway breakdown by threat level — and each latency exemplar links straight to its trace in Jaeger. Alert rules under `infra/prometheus/alerts.yml` fire on 5xx spikes, p95 regressions, security-violation floods, and any output-reflection event.

<p align="center">
  <img src="docs/grafana-traces-overview.png" alt="Grafana Trace Analytics — overview tiles and per-stage RED charts" width="760">
  <br>
  <img src="docs/grafana-traces-detail.png" alt="Grafana Trace Analytics — HTTP server spans and SecurityGateway verdicts by threat level" width="760">
  <br>
  <em>The Trace Analytics dashboard, built entirely from OpenTelemetry spans via the collector's spanmetrics connector: per-stage RED for the RAG pipeline (LLM generation dominates p95 at tens of seconds on CPU), HTTP routes, and SecurityGateway verdicts broken down by threat level.</em>
</p>

---

## Tech Stack

- **API & runtime** — FastAPI 0.115+, Uvicorn, Pydantic v2 (strict), `uv` package manager
- **RAG core** — ChromaDB · Ollama · `sentence-transformers/all-MiniLM-L6-v2` · pypdf · python-docx · markdown-it-py
- **Rate limiting** — in-process sliding window, or a shared **Redis** backend (same port, one config flag)
- **Security & quality** — Ruff (lint + format) · Mypy strict · `pip-audit` · Trivy · ~85% test coverage
- **Observability** — `structlog` (JSON) · `prometheus-client` + Grafana dashboards · **OpenTelemetry** traces via an OTel Collector (spanmetrics → Prometheus) to Jaeger
- **Frontend** — React 18 · Vite 5 · TypeScript (strict) · Tailwind CSS (demo console)
- **Delivery & infra** — Multi-stage Docker (non-root, CPU-only Torch, ~1.8 GB) · GitHub Actions CI · docker-compose · **AWS Terraform** (opt-in)

---

## Quick Start

```bash
git clone https://github.com/Carlosmarroquin20/Aegis-RAG.git
cd Aegis-RAG
cp .env.example .env
docker compose up -d
```

That single `docker compose up` starts the full stack — API, vector store, LLM, Prometheus, and a pre-loaded Grafana dashboard:

| Service | URL | Notes |
|---|---|---|
| **Web Console** | **`http://localhost:8080`** | **React demo UI (query · upload · security demo)** |
| Aegis-RAG API | `http://localhost:8000` | Main application |
| ChromaDB | `http://localhost:8001` | Vector store |
| Ollama | `http://localhost:11434` | Pulls `llama3.2` on first start |
| Prometheus | `http://localhost:9090` | Scrape + alerting |
| **Grafana** | **`http://localhost:3000`** | **Overview + Trace Analytics dashboards, anonymous viewer access** |
| Jaeger | `http://localhost:16686` | Distributed traces (OpenTelemetry) |
| OTel Collector | `http://localhost:8889/metrics` | Span-derived RED metrics (spanmetrics) |

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

## Web Console

A React + TypeScript demo UI ships with the stack at **http://localhost:8080**, with three tabs:

- **Query** — ask a question and see the grounded answer, the retrieved sources with relevance scores, and the threat level the gateway assigned.
- **Documents** — upload and index a TXT / MD / PDF / DOCX file.
- **Security demo** — fire curated prompt-injection payloads at the live gateway and watch which are blocked (with threat level and reason) before any retrieval or generation.

The `X-API-Key` is entered in the UI and kept only in the browser's `localStorage`. For frontend development, `cd frontend && npm run dev` runs Vite with a proxy to the API — see [`frontend/README.md`](frontend/README.md).

<p align="center">
  <img src="docs/web-console.png" alt="Aegis-RAG web console — query panel" width="720">
</p>

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
uv run pytest                       # unit + integration, 80% coverage gate (~85% actual)
uv run pytest tests/unit/           # fast, no I/O
uv run pytest tests/integration/    # in-memory adapters, ingestion pipeline
uv run pytest -m e2e --no-cov       # full HTTP surface against docker-compose
```

Over 200 tests span the security gateway, output sanitizer, rate limiter (both backends),
tracing spans, middleware, use cases, adapters and routes.

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

**CI pipeline** (GitHub Actions, every push — five parallel jobs):

```
Ruff (lint + format) · Mypy strict     ┐
Frontend build (tsc + Vite)            ├─→ all green required
pip-audit (dependency CVEs)            │
Unit + integration tests · 80% gate    │
Docker build · Trivy image scan        ┘
```

---

## Project Structure

```
src/aegis/
├── domain/           # Models · Ports · ChunkingService            (no I/O)
├── application/      # Use cases · DTOs                            (orchestration)
├── infrastructure/   # ChromaDB · Ollama · Parsers · Security      (adapters)
│   └── observability/  # Prometheus metrics · OpenTelemetry tracing
└── interface/api/    # FastAPI app · middleware · routes           (HTTP boundary)

infra/
├── prometheus/       # Scrape config + alert rules
├── grafana/          # Dashboard JSON + provisioning
└── terraform/        # AWS EC2 IaC — VPC + EC2 running the compose stack (opt-in)

tests/
├── unit/             # Pure-function tests + middleware + use case
├── integration/      # Use case ↔ adapter wiring with in-memory fakes
└── e2e/              # Full stack roundtrip (opt-in via -m e2e)

frontend/             # React + Vite + TS demo console (query · upload · security demo)
```

---

## License

MIT — see [LICENSE](LICENSE).
