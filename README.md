# Aegis-RAG

> **A RAG API in which security is part of the architecture, not a checklist.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/mypy-strict-1f5082)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OWASP LLM](https://img.shields.io/badge/OWASP_LLM_Top_10-mitigated-d6262c)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[![CI](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/codeql.yml/badge.svg)](https://github.com/Carlosmarroquin20/Aegis-RAG/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Carlosmarroquin20/Aegis-RAG/badge)](https://scorecard.dev/viewer/?uri=github.com/Carlosmarroquin20/Aegis-RAG)
[![Coverage](https://img.shields.io/badge/coverage-~94%25-brightgreen)](#testing)

Aegis-RAG is a Retrieval-Augmented Generation API designed for environments where you cannot afford to ship "best effort" security. Every query is evaluated by a domain-level **Security Gateway** before any retrieval or generation happens, every LLM response is post-processed by an **Output Sanitizer**, and the whole pipeline is wrapped in a defence-in-depth middleware stack with first-class observability.

The project also covers what running such a service involves: a hexagonal architecture with swappable adapters, strictly typed Python, mitigations mapped to the OWASP LLM Top 10, logs, metrics and distributed traces, a hardened multi-stage Docker build, unit, integration and end-to-end tests, AWS Terraform, and a React demo console.

## Results

Every figure below is measured, most of them in CI on every pull request, and each links to
how it was measured and what it misses.

| What | Measured on | Result |
|---|---|---|
| **Retrieval** (hybrid BM25 + dense, the default) | 141 labelled questions (56 in English), Spanish corpus | Right document in the top 5 for **96.5%** of questions; MRR **0.850** (dense alone: 0.813). Weakest on paraphrases. A CI gate. [ADR 0016](docs/adr/0016-hybrid-retrieval-by-default-reranker-opt-in.md), [ADR 0021](docs/adr/0021-larger-evaluation-sets.md) |
| **Answer faithfulness** (local NLI judge + figure check) | 271 labelled answers (half in English) | Catches **67%** of unfaithful answers at a **7.9%** false-flag rate (before the figure check and reply-word fix: 55% and 12.5%). Weakest on contradictions (19 of 35). A CI gate. [ADR 0017](docs/adr/0017-answer-faithfulness-judged-by-nli.md), [ADR 0022](docs/adr/0022-faithfulness-reply-words-and-figure-check.md) |
| **Input gateway** (signatures, English and Spanish) | 326 labelled queries | Blocks **77.5%** of the attacks it targets and **60.4%** of all attacks, at **6.4%** false positives. [Report](docs/GATEWAY_EVALUATION.md) |
| **ML injection classifier** (not enforcing) | the same 326 queries | Would block **92.9%** of attacks, but **17.4%** of legitimate queries; available in shadow mode, which only records. [ADR 0019](docs/adr/0019-ml-injection-classifier-measured-not-enabled.md), [ADR 0020](docs/adr/0020-injection-classifier-in-shadow-mode.md) |
| **Throughput** (one worker, k6) | compose stack | About **70 requests/s**; p99 under **75 ms** at a steady 40/s. [Load tests](loadtest/README.md) |
| **Tests** | unit, integration, end-to-end | About 1,000 tests, ~94% coverage, `mypy --strict` |

<p align="center">
  <img src="docs/security-demo.gif" alt="Aegis-RAG web console — prompt-injection probes blocked by the live gateway" width="720">
  <br>
  <em>The web console's security demo: fire prompt-injection payloads at the live gateway and watch them get blocked before any retrieval happens.</em>
</p>

---

## Engineering Highlights

- **Defence in depth, by design.** The ASGI middleware chain (RequestID → AccessLog → SecurityHeaders → CORS → APIKey → RateLimit → ErrorBoundary → BodyLimit) executes *before* any route handler, so 404s and 405s receive the same hardening as authenticated traffic. RequestID sits outermost, so every log line and span carries the same correlation ID; ErrorBoundary sits innermost, so even an unhandled 500 is logged, counted and returned with security headers instead of bypassing the stack.
- **Bounded label cardinality.** Prometheus path labels collapse to the matched FastAPI route template (`/api/v1/documents/{doc_id}`), never the raw URL, so the number of time series stays bounded however many distinct URLs are requested.
- **Detection that does not become the attack surface.** The gateway inspects attacker-controlled text on the event loop, so every signature is kept linear on non-matching input and a regression suite feeds each rule its worst-case 8 KB payload. Obfuscation that leaves a payload readable — zero-width characters, fullwidth forms, bidi overrides, Cyrillic/Greek homoglyphs — is normalized or folded before matching.
- **Measured, not asserted.** The gateway is scored against a [labelled corpus of 326 queries](docs/GATEWAY_EVALUATION.md) that includes the attack classes it does *not* claim to stop. In its default mode it blocks **78%** of the attacks it targets — in English and Spanish — and **60%** of the whole corpus, at a **6%** false-positive rate that falls entirely on deliberately adversarial queries — no everyday question is blocked. The report is generated, lists every miss, and a test fails if it goes stale. The first run found two real defects, both fixed.
- **Indirect prompt injection caught at the exit.** The `OutputSanitizer` blocks LLM responses that *echo* injection payloads from poisoned documents. The use case deliberately does not catch the error so it surfaces as a pipeline failure (HTTP 500), not a silent leak.
- **Hybrid retrieval, chosen by measurement.** Dense similarity is fused with BM25 keyword search (reciprocal rank fusion). On 141 labelled questions this raises MRR from 0.813 to 0.850 at no measurable cost per query. The gain is all on Spanish questions; BM25 cannot help English questions over a Spanish corpus, and the evaluation reports that separately. A multilingual cross-encoder reranker reaches 0.905, but on CPU it costs about 0.8 s per query and puts only 2 more questions into the five chunks the model sees, so it is [opt-in](docs/adr/0016-hybrid-retrieval-by-default-reranker-opt-in.md). The same evaluation runs in CI against the application's own retrieval code and fails the build if quality drops.
- **Answers checked against their sources.** A local multilingual NLI model checks each sentence of an answer against the retrieved context and names the ones it does not support: no LLM judge, no API calls, the same verdict every run. A sentence that states a figure the context never mentions is flagged whatever the model thinks, because NLI models read "60 days" as entailed by "30 days". On 271 labelled answers, half of them in English, the judge catches 67% of unfaithful answers at a 7.9% false-flag rate, up from 55% and 12.5% before those [two measured fixes](docs/adr/0022-faithfulness-reply-words-and-figure-check.md). CI gates on it; in production it is an opt-in score and alert.
- **Streaming without bypassing the output guard.** `POST /api/v1/query/stream` sends the answer as it is generated, but only in whole sentences that have already passed the sanitizer, with an 80-character margin scanned before each one is released. A reflected injection is never sent, even in part. Refusals before the first text keep their status codes; failures after it end the stream with an `error` event the client must honour ([ADR 0018](docs/adr/0018-stream-whole-sanitized-sentences.md)).
- **Magic-byte file detection.** Document uploads are dispatched to parsers based on actual file content, not the attacker-controllable `Content-Type` header.
- **Hexagonal for real.** The `LLMClientPort`, `VectorStorePort` and `RateLimitStore` interfaces mean swapping Ollama for OpenAI, ChromaDB for pgvector, or the in-memory rate limiter for the shared **Redis** backend is a one-file change — and there are fakes in the test suite that prove it. The dependency rule itself (`interface → application → domain ← infrastructure`) is enforced by a test that parses every module's imports. Adapters translate vendor failures at the boundary, so a vector-store or LLM outage reaches the client as a retryable **503/504**, never as a leaked client-library exception.
- **Load-tested, with the findings fixed.** [k6 scenarios](loadtest/README.md) against the compose stack found three defects the unit tests could not: a rate-limit check whose cost grew with the limit (56% of CPU), garbage-collection pauses of 1.3 s that were the p99, and a synchronous Redis call that let a Redis stall freeze `/health`. Fixed, each with a regression test: the ceiling on one worker went from 37 to about 70 requests per second and the p99 at a steady 40/s from 974 ms to under 75 ms.
- **A supply chain you can verify.** Every action is pinned by commit and every base image by digest; CodeQL and OpenSSF Scorecard run on the repository itself. Released images are signed keylessly with Sigstore and carry SLSA build provenance and an SPDX SBOM, so a consumer can check what they run came from this repository's workflow and from which commit.
- **Three pillars of observability.** Structured JSON logs, Prometheus metrics, and opt-in **OpenTelemetry** traces (a span per pipeline stage) all correlate through a single `request_id` / `trace_id`.
- **Air-gap by default, hosted by choice.** Out of the box the full stack runs on local infrastructure (Ollama + ChromaDB) and no data leaves the network. The same `LLMClientPort` also has an adapter for any OpenAI-compatible hosted provider (Groq, OpenRouter, Mistral, a vLLM gateway), selected with one setting — an explicit opt-in that is logged at startup, because it is the one configuration in which document content leaves the host.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Interface     FastAPI routes + middleware stack                     │
│                RequestID → AccessLog → SecurityHeaders → CORS →      │
│                APIKey → RateLimit → ErrorBoundary → BodyLimit →      │
│                Routes                                                │
├──────────────────────────────────────────────────────────────────────┤
│  Application   Use cases (QueryRAG, IngestDocuments) · Retriever     │
├──────────────────────────────────────────────────────────────────────┤
│  Domain        Models · Ports · ChunkingService · RankFusion         │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure  ChromaDB · BM25 · Reranker · Ollama · Parsers       │
│                  SecurityGateway                                     │
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
    A->>O: sanitize(answer) · LLM05
    O-->>A: clean
    A-->>C: 200 JSON {answer, sources}
    Note over A,G: a BLOCKED verdict short-circuits to HTTP 400 — no retrieval, no LLM call
```

> **▶ [Open the interactive sequence](https://carlosmarroquin20.github.io/Aegis-RAG/aegis-query-sequence.html)** — the same security-first flow as an explorable diagram with activation bars and phase bands, as a [JSON spec](docs/diagrams/aegis-query-sequence.json) (diagram-as-code, rendered with Archify).

Document ingestion follows the same hexagonal pattern: magic-byte MIME detection → parser dispatch (TXT/MD/PDF/DOCX) → `ChunkingService` (paragraph → sentence → word fallback with overlap stitching) → content-addressed deduplication → vector store upsert.

---

## Security Controls — OWASP LLM Top 10 (2025)

> The full **[threat model](docs/THREAT_MODEL.md)** lists 26 threats across the query, transport, ingestion, supply-chain and tenancy surfaces — each with the module that enforces the control and the test that verifies it — and states the residual risks plainly, including what the input gateway does *not* catch.
>
> How much the input gateway actually catches is **[measured](docs/GATEWAY_EVALUATION.md)**, per attack category, with the individual misses and false positives listed.
>
> The reasoning behind the non-obvious choices — why the rate limiter fails open, why an auth lockout also refuses valid keys, why `X-Forwarded-For` is ignored by default — is recorded as **[architecture decision records](docs/adr/README.md)**, each with the alternatives that were rejected.

| Threat | Control | OWASP |
|---|---|---|
| Prompt injection | Regex signatures in English and Spanish (voseo included) + staged-reasoning (CoT) check + Shannon entropy, matched after Unicode NFKC normalization, invisible-character stripping, accent removal and homoglyph folding | LLM01 |
| Improper output handling | Length cap · HTML strip · reflection detection · PII flagging | LLM05 |
| System-prompt leakage | Signature rules block probing queries (*"show me your prompt"*) | LLM07 |
| API abuse | Per-tenant sliding-window rate limiter with burst allowance · rejected requests never occupy the window · optional daily LLM token budget per tenant · explicit fail-open/fail-closed policy if the store is down | LLM10 |
| Cross-tenant access | Tenant resolved from the API key alone · one collection per tenant, no fallback to a default · a tenants file that shares a key or a collection stops startup | LLM02, LLM08 |
| Credential guessing | Client address locked out (429) after repeated rejected keys — valid keys included, so the response is no oracle · address taken from `X-Forwarded-For` only at an explicitly trusted proxy depth (`TRUSTED_PROXY_HOPS`, default 0) · short keys reported at startup | General |
| Sensitive data exposure | API key auth at ASGI level (constant-time digest comparison; the key never reaches logs or the rate-limit store) · non-root container · `Cache-Control: no-store` · OpenAPI schema and docs disabled outside debug | LLM02 |
| File-upload attacks | Magic-byte MIME detection · request body capped at the ASGI layer as it streams in (no reliance on a proxy; chunked uploads included) · 50 MB file cap enforced before the upload is buffered · uncompressed-size budget for DOCX archives (decompression bombs) | General |
| Silently wrong retrieval | Each vector collection records the embedding model that built it; the API refuses to start on a mismatch, which ChromaDB itself accepts in silence | LLM08 |
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
| `aegis_auth_failures_total` | Counter | Requests refused at authentication, by outcome (`rejected` / `throttled`) |
| `aegis_rate_limit_backend_errors_total` | Counter | Requests served or refused while the rate-limit store was unreachable |
| `aegis_stream_first_delta_seconds` | Histogram | Time to the first sanitized text of a streamed answer |
| `aegis_streams_total` | Counter | Streamed answers by outcome: `completed`, `blocked`, `failed`, `disconnected` |
| `aegis_injection_shadow_total` | Counter | Allowed queries by shadow-classifier outcome: `flagged`, `clean`, `busy`, `error` |
| `aegis_answer_faithfulness` | Histogram | Share of an answer's claims its context supports (`FAITHFULNESS_CHECK`) |
| `aegis_answers_judged_total` | Counter | Judged answers by outcome: `supported`, `unsupported`, `abstained` |

Logs are single-line JSON via `structlog` with an auto-bound `request_id` for end-to-end correlation between logs, metrics, traces, and the `X-Request-ID` response header.

**Distributed tracing (OpenTelemetry)** is opt-in (`TRACING_ENABLED`). When on, the FastAPI server span and the HTTPX call to Ollama are auto-instrumented, the query use case emits a child span per pipeline stage (`security.evaluate` → `rag.retrieve` → `llm.generate` → `output.sanitize`, with `rag.retrieve` split into `retrieve.dense`, `retrieve.lexical`, `retrieve.fuse` and `retrieve.rerank`), and every log line carries the active `trace_id`/`span_id` so logs pivot to their trace. Spans export over OTLP/HTTP to an **OpenTelemetry Collector**, which fans them out two ways: the raw traces to a Jaeger backend at `http://localhost:16686`, and — via the **spanmetrics connector** — trace-derived RED metrics to Prometheus. The SDK lives in the optional `otel` extra, so the base image stays lean when tracing is off.

Two pre-provisioned Grafana dashboards ship with the stack. **Production Overview** surfaces request rate, error rate, p95 latency, and blocked queries as headline tiles, then breaks down HTTP and Security into full-resolution time series, followed by LLM usage and cost, **Streaming** (time to first text, and streams completed, blocked mid-answer or abandoned) **Answer faithfulness** (share of unsupported answers, mean score, abstentions, judge failures, score distribution) and the **injection classifier in shadow mode** (flag rate, skipped and failed scores, score distribution). **Trace Analytics** turns the pipeline spans into per-stage RED charts (call rate, errors, and p50/p95/p99 latency for `security.evaluate` / `rag.retrieve` / `llm.generate` / `output.sanitize` / `output.faithfulness`), the retrieval stages inside `rag.retrieve` (dense, lexical, fusion, rerank), plus a SecurityGateway breakdown by threat level — and each latency exemplar links straight to its trace in Jaeger. A unit test checks that every metric a panel or an alert queries is exported and every span the trace dashboard filters on is opened by the code, so a rename cannot leave a panel quietly reading "No data". Alert rules under `infra/prometheus/alerts.yml` fire on 5xx spikes, p95 regressions, security-violation floods, any output-reflection event, sustained authentication failures, an unreachable rate-limit store, an LLM backend whose circuit breaker stays open, a spent tenant budget, and a high share of answers the faithfulness check cannot support.

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
- **RAG core** — ChromaDB · Ollama (local) or any OpenAI-compatible hosted LLM · `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual) · BM25 + reciprocal rank fusion · optional `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranker · pypdf · python-docx · markdown-it-py
- **Rate limiting** — in-process sliding window, or a shared **Redis** backend (same port, one config flag)
- **Security & quality** — Ruff (lint + format) · Mypy strict · `pip-audit` · Trivy · ~90% test coverage
- **Observability** — `structlog` (JSON) · `prometheus-client` + Grafana dashboards · **OpenTelemetry** traces via an OTel Collector (spanmetrics → Prometheus) to Jaeger
- **Frontend** — React 19 · Vite 8 · TypeScript (strict) · Tailwind CSS (demo console)
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

Or stream the answer as it is generated (server-sent events: `start`, `delta`, then `end` with
the complete answer):

```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "What is the security model?"}'
```

Or run the same flow as a single test: `uv run pytest -m e2e --no-cov`.

**Try it on a realistic domain.** [`examples/atencion-cliente`](examples/README.md) is a Spanish
customer-service corpus for a fictional retailer — returns, warranties, shipping, invoicing,
loyalty programme. Seed it and ask in Spanish; the guide lists questions it answers and attacks
the gateway refuses. Nothing in the code is specific to it: a different domain is a different
directory.

```bash
uv run python scripts/seed_documents.py --source-dir examples/atencion-cliente
```

**Prebuilt images.** Every release publishes the API and the console to GHCR as
`ghcr.io/carlosmarroquin20/aegis-rag-api` and `ghcr.io/carlosmarroquin20/aegis-rag-frontend`,
signed with Sigstore and shipped with build provenance and an SPDX SBOM. See
[Supply Chain](SECURITY.md#supply-chain) for how to verify one before running it.

---

## Retrieval

Retrieval finds the chunks the LLM answers from, and it decides whether the model ever sees the
answer. `RETRIEVAL_MODE` picks the stages:

| `RETRIEVAL_MODE` | Stages | MRR | hit@5 | Per query (CPU) |
|---|---|---|---|---|
| `dense` | Vector similarity | 0.813 | 95.7% | 16 ms |
| `hybrid` *(default)* | Vector similarity + BM25, fused by reciprocal rank fusion | 0.850 | 96.5% | 16 ms |
| `dense+rerank` / `hybrid+rerank` | The above, re-scored by a multilingual cross-encoder | 0.899 / 0.905 | 97.2% / 97.9% | ~0.8 s |

Figures are for 141 labelled questions over [`examples/atencion-cliente`](examples/README.md),
measured in CI. 56 of the questions are in English, and each question is tagged as direct,
paraphrase or scenario. The report breaks the figures down both ways
([ADR 0021](docs/adr/0021-larger-evaluation-sets.md)): paraphrases are the hardest kind (hit@5 90.3%). Dense embeddings match meaning but blur exact terms (a product, a figure, a rare
word); BM25 matches exactly those and misses paraphrases and other languages, so the fused ranking
recovers what either alone drops. The keyword index is kept in memory per collection and rebuilt
from the vector store when its contents change, so it never becomes a second source of truth.

The reranker is off by default. It puts the right document first for 12 more questions, but the
model reads the top five chunks, and there it gains 2 questions of 141. That is not worth about
0.8 s per query on CPU. To turn it on, set both in `.env` and rebuild, so the model is
baked into the image and loads offline:

```bash
RETRIEVAL_MODE=hybrid+rerank
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Compare the modes on your own corpus and questions:

```bash
uv run python scripts/evaluate_retrieval.py --compare --corpus <dir> --questions <file.jsonl>
```

The reasoning and the trade-offs are in [ADR 0016](docs/adr/0016-hybrid-retrieval-by-default-reranker-opt-in.md).

---

## Answer faithfulness

Retrieval decides whether the model sees the answer; faithfulness is whether it then sticks to it.
Each answer is split into sentences, and each sentence is checked against the sentences of the
retrieved context by a local, multilingual NLI model
(`MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli`). A sentence is supported when some part of
the context entails it, net of contradiction, and it states no figure the context or the question
does not. Before judging, a leading "Yes,"/"No," that only answers the question is dropped. An
answer that declines for lack of context is counted as an abstention, not a failure.

Measured on 271 labelled answers over the example corpus, 136 of them in English
(`scripts/evaluate_faithfulness.py`):

| Judge | Unfaithful caught | Faithful flagged |
|---|---|---|
| NLI over context sentences | 54.6% | 12.5% |
| + reply words dropped | 55.5% | 7.9% |
| + figure check | 67.2% | 12.5% |
| **Both** *(shipped)* | **67.2%** | **7.9%** |

Sentence premises were chosen on the first 80 of these answers, where they beat whole chunks by
65% to 43% ([ADR 0017](docs/adr/0017-answer-faithfulness-judged-by-nli.md)). The two corrections
came from the larger set ([ADR 0022](docs/adr/0022-faithfulness-reply-words-and-figure-check.md)). Most false flags were answers opening with a bare
"No,", and half the missed figures appeared nowhere in the source. With both, the judge catches
15 of 17 invented additions, 13 of 16 outside-knowledge claims and 33 of 51 changed figures.
Contradictions that reuse the context's own words are now its weak spot, at 19 of 35. The flagged
faithful answers are mostly closing courtesies, which no context supports. CI fails if recall drops
below 0.66 or false flags rise above 0.09. To measure your own deployment's LLM end to end, run the
live mode against a running stack:

```bash
uv run python scripts/evaluate_faithfulness.py --live
```

In production the check is opt-in. Set `FAITHFULNESS_CHECK=true` and `FAITHFULNESS_MODEL` in
`.env`, then rebuild so the model is baked into the image. Every response then carries a
`faithfulness` field:

```json
"faithfulness": {"score": 0.5, "abstained": false, "unsupported": ["La tienda paga el flete."]}
```

The scores feed `aegis_answer_faithfulness`, and `AegisUnsupportedAnswersHigh` fires when over a
quarter of answers are flagged. The answer itself is never withheld. Why NLI rather than an LLM
judge, and what the measurements showed along the way, is in
[ADR 0017](docs/adr/0017-answer-faithfulness-judged-by-nli.md).

---

## Choosing the LLM backend

The model sits behind `LLMClientPort`, and the backend is a configuration choice:

| `LLM_PROVIDER` | Backend | Data leaves the host? | Typical latency |
|---|---|---|---|
| `ollama` *(default)* | Local Ollama | No — air-gapped | Tens of seconds on CPU |
| `openai_compatible` | Any provider implementing `POST /chat/completions` | **Yes** — the query and the retrieved excerpts | Around a second |

To use a hosted provider, put these in `.env` (git-ignored) and restart:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.groq.com/openai/v1     # or OpenRouter, Mistral, your own gateway
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=<your key>
```

Nothing else changes. The SecurityGateway still runs before retrieval, the same hardened system
prompt is sent, and the OutputSanitizer still post-processes the answer; a provider outage, a
rejected key or a rate limit surfaces as a **503**, and a timeout as a **504**. The key is held as a
`SecretStr`, sent only in the `Authorization` header, and never written to logs or error messages.
The app refuses to start if the hosted provider is selected without its settings, or with a
non-HTTPS URL outside debug mode.

**Fallback.** A hosted provider's rate limits and outages need not reach users. Name a second
backend and it answers when the first is rate limited, down or too slow:

```bash
LLM_FALLBACK_PROVIDER=ollama    # hosted primary, local model behind it
```

Each backend sits behind a circuit breaker: three consecutive failures open it, it is skipped for
30 seconds, then one trial request decides whether it closes. A rate-limited reply opens it at
once for as long as the provider's `Retry-After` asks. Answers keep reporting the backend that
produced them, so cost is attributed correctly and a local answer is never billed at the hosted
rate. The reasoning, including what the order means for where data goes, is in
[ADR 0009](docs/adr/0009-llm-fallback-behind-circuit-breakers.md).

**Usage and cost.** Every answer records the tokens it consumed, per provider, model and client
(the digest the API assigns to each key, never the key itself). Set the provider's prices and the
estimated spend follows:

```bash
LLM_INPUT_PRICE_PER_MTOK=0.15    # US dollars per million input tokens
LLM_OUTPUT_PRICE_PER_MTOK=0.60   # and per million output tokens
```

The *LLM usage and cost* row of the Production Overview dashboard shows spend, tokens, cost per
answered query and spend by client and model. A backend that reports no usage is counted
separately (`aegis_llm_usage_unreported_total`), so missing data shows up instead of reading as
free. The `llm.generate` span carries the OpenTelemetry GenAI attributes (`gen_ai.usage.*`).

---

## Multi-tenancy

One deployment can serve several teams, each with its own documents, request limit and daily
LLM token budget. Declare them in a TOML file and point `TENANTS_FILE` at it:

```toml
[[tenant]]
id = "support"
collection = "support-docs"
api_key_sha256 = ["<digest from scripts/tenant_key.py>"]
rate_limit_requests = 120
daily_token_budget = 2_000_000
```

- **The key decides the tenant.** It is resolved before any route runs; nothing a caller sends
  can name another tenant.
- **Each tenant has its own collection,** and every route reaches only that one: a tenant cannot
  list, retrieve, delete or be answered from another's documents.
- **The file holds key digests, not keys,** so it is configuration, not a secret. A tenant can
  hold several keys, which makes rotation a config change.
- **Limits are per tenant.** All of a tenant's keys share one rate-limit bucket. A spent token
  budget answers `429` with a `Retry-After` until 00:00 UTC; budgets live in Redis, so they hold
  across replicas.
- **Cost per team is a query:** the usage and cost metrics are labelled by tenant, and
  `AegisTenantTokenBudgetExhausted` tells the platform team when one is being refused.

`scripts/tenant_key.py` generates a key and its digest; `examples/tenants.example.toml` is a
commented template; `scripts/seed_documents.py --tenant support` indexes into one tenant's
collection. A file that would leak data (two tenants on one key or one collection) or lift a
limit by accident (a misspelt field) stops the API at startup. Without `TENANTS_FILE`, every key
in `VALID_API_KEYS` behaves exactly as before. The design and its trade-offs are in
[ADR 0010](docs/adr/0010-tenants-isolated-by-collection.md).

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
| `GET`  | `/ready`  | Readiness probe (checks ChromaDB + Ollama; 503 until both answer) |
| `GET`  | `/metrics` | Prometheus scrape target |
| `POST` | `/api/v1/query` | Submit a question to the RAG pipeline |
| `POST` | `/api/v1/documents` | Upload and index a document (TXT, MD, PDF, DOCX) |
| `GET`  | `/api/v1/documents` | List indexed chunks (paginated) |
| `DELETE` | `/api/v1/documents/{id}` | Delete a single chunk |
| `DELETE` | `/api/v1/documents` | Bulk delete by ID list (max 500) |

Interactive Swagger docs are served at `http://localhost:8000/docs` when `DEBUG=true`. Outside debug mode `/docs`, `/redoc` and `/openapi.json` are not served at all.

Error semantics are part of the contract: a blocked query is a structured **400**, a throttled client gets **429** with an accurate `Retry-After` (as does an address locked out after repeated bad keys), an unavailable vector store or LLM is **503** (or **504** on an LLM timeout), and only a genuine defect is a **500**.

---

## Testing

The suite is layered so each level catches what the others cannot.

```bash
uv run pytest                       # unit + integration, 80% coverage gate (~90% actual)
uv run pytest tests/unit/           # fast, no I/O
uv run pytest tests/integration/    # in-memory adapters, ingestion pipeline
uv run pytest -m e2e --no-cov       # full HTTP surface against docker-compose
```

Over 550 tests span the security gateway, output sanitizer, rate limiter (both backends),
tracing spans, middleware, use cases, adapters and routes. They include regression suites for
the failure modes that matter most here: obfuscated injection payloads, worst-case regex
inputs, retry storms against the rate limiter, and dependency outages.

**Gateway evaluation.** Detection and false-positive rates are measured rather than inferred
from the unit tests, which only show that chosen examples behave as intended:

```bash
uv run python scripts/evaluate_gateway.py            # rates per mode, plus latency
uv run python scripts/evaluate_gateway.py --write    # regenerate docs/GATEWAY_EVALUATION.md
uv run python scripts/evaluate_gateway.py --dataset other.jsonl   # score a third-party corpus
```

**End-to-end tests** issue real requests through the live middleware stack. They auto-skip with a clear message if `docker compose` is not up, so CI never fails on a forgotten container. Point them at any environment via env vars:

```bash
AEGIS_E2E_BASE_URL=https://staging.example.com \
AEGIS_E2E_API_KEY=<key> uv run pytest -m e2e --no-cov
```

**Lint, format, type-check:**

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
uv run mypy src/ scripts/
```

**Before committing**, the same checks run locally through [pre-commit](https://pre-commit.com/)
(ruff, mypy, lockfile consistency, file hygiene and `gitleaks` secret scanning):

```bash
uv run pre-commit install             # once per clone
uv run pre-commit run --all-files     # on demand
```

The linter, formatter and type checker in the hooks come from `uv.lock`, so a hook and the
pipeline cannot disagree about what "clean" means.

**CI pipeline** (GitHub Actions, every push and pull request — eight jobs):

```
Ruff (lint + format) · Mypy strict              ┐
Unit + integration tests · 80% coverage gate    │
pip-audit (dependency CVEs)                     │
Docker build · Trivy image scan                 ├─→ all green required
Frontend build · image build · nginx -t         │
Terraform fmt + validate                        │
Compose stack smoke test · tests on real Redis  │
Secret scan of the full git history (gitleaks)  ┘
```

The last three exist so that dependency updates are verified rather than assumed: a base-image,
provider or stack-image bump is built, validated or started before it can merge. Dependabot opens
those updates weekly.

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

evals/gateway/        # Labelled attack / legitimate-query corpus for the gateway
examples/             # Ready-to-seed demo corpora (Spanish customer service)
scripts/              # Corpus seeding · gateway evaluation

frontend/             # React + Vite + TS demo console (query · upload · security demo)
```

---

## Changelog

Release notes are in [`CHANGELOG.md`](CHANGELOG.md). The current release is **1.3.0**.

---

## License

MIT — see [LICENSE](LICENSE).
