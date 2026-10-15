# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries describe what changed for someone running or integrating the service, not the
commit history; the reasoning behind the larger decisions is in [`docs/adr`](docs/adr/README.md).

## [Unreleased]

### Added

- **Spanish prompt-injection signatures**: instruction override, prompt disclosure, persona
  hijack, roleplay and staged-reasoning attacks written in Spanish, voseo included, are now
  blocked. Signatures are matched with accents removed, so unaccented spellings are covered.
- The gateway corpus gains a Spanish set (34 attacks, 49 legitimate queries) that was committed
  before the Spanish rules. Measured on it: 33 of 34 attacks blocked, no legitimate query
  blocked.

### Fixed

- A combining mark placed inside an English keyword (`i̇gnore`) hid it from the
  signatures.

## [1.0.1] — 2026-10-13

Maintenance release: no change to the API contract. It adds a measurement of the input
gateway, fixes the two detector defects that measurement found, and brings every runtime
dependency up to date.

### Added

- **Gateway evaluation.** A labelled corpus of 242 queries (`evals/gateway/dataset.jsonl`)
  and `scripts/evaluate_gateway.py`, which reports detection and false-positive rates per
  category with confidence intervals. The generated report is
  [`docs/GATEWAY_EVALUATION.md`](docs/GATEWAY_EVALUATION.md).
- Pre-commit hooks mirroring the CI gates, and a secret scan of the full history in CI.
- Dependabot for every ecosystem in the repository, and CI jobs that build the frontend
  image, validate the Terraform configuration and smoke-test the compose stack, so a
  dependency update is exercised before it is merged.

### Changed

- Observability stack upgraded: Prometheus 2.55 → 3.15, Grafana 11.3 → 13.2, OpenTelemetry
  Collector 0.115 → 0.162, Jaeger 1.62 → 1.76. Dashboards, alert rules, span metrics and
  exemplar links were verified against the running stack. The collector configuration uses
  the current component names (`span_metrics`, `otlp_grpc`); metric names are unchanged.
- Base images and providers updated: Redis 7 → 8, nginx 1.27 → 1.31, Node 20 → 26 for the
  frontend build, TypeScript 5.9 → 7.0, the AWS Terraform provider, and the Python
  dependency set.

### Fixed

- **Long Japanese and Chinese questions were rejected in strict mode** as encoded payloads.
  The entropy check now measures ASCII characters only.
- **"You are now an …" persona hijacks were not detected**; the signature accepted only the
  article "a".

## [1.0.0] — 2026-10-12

First stable release. The API contract, the error semantics and the security model are now
documented and covered by tests, and the full stack has been exercised end to end.

The bulk of this release is a security and correctness review of the 0.1.0 codebase. Every
item under *Security* and *Fixed* was reproduced before it was changed and has a regression
test.

### Security

- **Denial of service through the input gateway.** Three injection signatures backtracked
  for about a second on a single 8 KB query, stalling the event loop. All patterns are now
  bounded or anchored; worst-case evaluation is about 20 ms.
- **Unicode obfuscation bypassed the gateway.** Zero-width characters, fullwidth forms, bidi
  overrides and Cyrillic/Greek homoglyphs let a blocked payload through with a score of 0.
  Queries are NFKC-normalized, stripped of format characters and matched on a
  homoglyph-folded view.
- **The OpenAPI schema was served without authentication in production.** It is now
  debug-only, together with the docs pages.
- **API keys were not compared in constant time**, were used verbatim as Redis key names,
  and had their first eight characters written to logs. Keys are compared as digests with
  `hmac.compare_digest`; downstream code only ever sees a truncated digest.
- **No limit on authentication attempts.** A client address is locked out after repeated
  rejected keys. The lockout also refuses valid keys, so it is not a guessing oracle.
- **Decompression bombs.** A DOCX is refused before opening if it would inflate past a size
  budget; the 50 MB upload limit only bounded compressed bytes.
- **The upload limit was checked after the file was already in memory**, and nothing bounded
  the request body itself. The file is now size-checked before buffering, and the body is
  capped at the ASGI layer as it streams in, including chunked uploads.
- **Unhandled errors bypassed the middleware stack.** A 500 carried no security headers or
  request ID and was missing from the error-rate metric, so the alert built on it could not
  see them.
- **`X-Request-ID` was reflected and logged unvalidated.**
- **Upload filenames were stored verbatim**, accepting path components and the
  right-to-left override used to disguise file extensions.
- **The compose stack published every port on all interfaces**, exposing unauthenticated
  ChromaDB and Ollama, Prometheus's lifecycle endpoint and Grafana's default credentials to
  the local network. Ports are bound to loopback.
- **CI supply chain.** The image scanner ran from a mutable branch reference; it and the
  toolchain installer are pinned to commit SHAs, and the job token is read-only.
- Dependencies with newly published advisories were upgraded (`pypdf`, `urllib3`,
  `multidict`, `oauthlib`, `virtualenv`).

### Added

- **Hosted LLM backends.** `LLM_PROVIDER=openai_compatible` selects any provider that
  implements the OpenAI Chat Completions API (Groq, OpenRouter, Mistral, a vLLM gateway).
  Opt-in, logged at startup, and off by default so the stack remains air-gapped.
- **Plain-text answers.** Markdown emphasis and headings are removed from model output and
  links are rewritten as `text (target)` so a label cannot hide its destination.
- **Explicit behaviour when the rate-limit store is down** (`RATE_LIMIT_FAIL_OPEN`), with a
  metric and an alert in both modes.
- `503` with `Retry-After` when the vector store or LLM is unavailable, `504` on an LLM
  timeout, instead of a generic `500`.
- Metrics `aegis_auth_failures_total` and `aegis_rate_limit_backend_errors_total`, and the
  alerts `AegisAuthFailureSpike` and `AegisRateLimitBackendDown`.
- A [threat model](docs/THREAT_MODEL.md) and [architecture decision records](docs/adr/README.md),
  each kept consistent with the code by a test.
- A test that enforces the layer dependency rule by parsing every module's imports.
- Startup warnings when no API key is configured or a configured key is short.

### Changed

- **`collection` is no longer accepted** by the query and ingest endpoints. It was ignored,
  and the ingest response echoed the requested name while writing to the default collection.
  The response now always reports the collection actually written to.
- **The 400 returned for a blocked query is documented as it is sent**: the fields are nested
  under `detail`. The OpenAPI schema previously described a flat body the endpoint never
  produced.
- `X-RateLimit-Limit` reports the effective ceiling including burst, so it agrees with
  `X-RateLimit-Remaining`.
- Requests rejected by the rate limiter no longer occupy the window. `Retry-After` is now
  accurate, and a client that waits it is admitted.
- The middleware chain gained two inner layers:
  `… → RateLimit → ErrorBoundary → BodyLimit → routes`.
- The application layer no longer imports from infrastructure; the query use case depends
  on ports for the security guards and metrics.
- ChromaDB and parser calls run off the event loop.
- The version is read from the installed package instead of being repeated in settings.

### Fixed

- Two gateway rules blocked ordinary questions: any query naming someone called "Dan", and
  any "show/display … context/instructions" request over the corpus.
- HTML stripping deleted text between comparison operators (`3 < 5 and 7 > 2`).
- Markdown files were indexed with HTML entities in place of `&`, `<`, `>` and quotes;
  tables were indexed as rows of pipes and dashes.
- `X-RateLimit-Reset` reported a timestamp decades in the future; `Retry-After` could be `0`.
- Deduplication on re-ingestion only worked against ChromaDB and was untested; it now goes
  through the vector-store port.
- The seed script crashed on startup and used a different chunker and ID scheme from the API.
- The docs pages rendered blank in debug mode under the strict Content-Security-Policy.
- The frontend proxy rejected uploads over 1 MB and timed out before the LLM did.
- An oversized streamed upload was cut off correctly but reported as `400` instead of `413`.
- The application failed to start when `.env` contained a variable it does not model.
- The compose stack could not start alongside a natively installed Ollama.

## [0.1.0] — 2026-09-08

Initial public version: hexagonal RAG API with the SecurityGateway and OutputSanitizer,
document ingestion (TXT, Markdown, PDF, DOCX), API-key authentication, in-memory and Redis
rate limiting, structured logging, Prometheus metrics, OpenTelemetry tracing with Grafana
dashboards, a React demo console, a multi-stage Docker image, docker-compose stack, GitHub
Actions CI and an AWS Terraform skeleton.

[1.0.1]: https://github.com/Carlosmarroquin20/Aegis-RAG/releases/tag/v1.0.1
[1.0.0]: https://github.com/Carlosmarroquin20/Aegis-RAG/releases/tag/v1.0.0
[0.1.0]: https://github.com/Carlosmarroquin20/Aegis-RAG/tree/f653d4e
