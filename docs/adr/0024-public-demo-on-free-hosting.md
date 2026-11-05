# 0024 — A public demo on free hosting: one container, a public key, a read-only corpus

**Status:** Accepted

## Context

The project is meant to be looked at, and almost nobody who looks at it will clone the
repository and start eight containers. A hosted demo removes that step, with two constraints:
it must cost nothing, and the security properties it demonstrates must survive being open to
anyone on the internet.

Free hosting rules out most of the compose stack. The platforms that fit (Hugging Face Spaces
on its free CPU tier) run a single container, with no GPU and no persistent services beside
it. A local model on that hardware takes the better part of a minute per answer, and there is
nowhere to run Chroma, Redis, Prometheus, Grafana or Jaeger.

## Decision

Build a separate image for the demo (`deploy/huggingface/`) rather than adapting the
production one:

- **One container.** nginx serves the console and forwards the API paths to uvicorn on
  loopback; Chroma runs in the API process (`CHROMA_MODE=embedded`). Only the API and
  readiness paths are forwarded, so `/metrics` is not reachable from outside.
- **The corpus is indexed at build time**, through the same ingestion pipeline as an upload,
  into the collection of the demo tenant. The container starts with its documents and never
  has to fetch or write them.
- **Read-only.** `DOCUMENT_WRITES_ENABLED` is off, and nginx refuses write methods on the
  documents route as well. Ingested documents are trusted and skip the gateway
  ([0003](0003-documents-trusted-at-ingestion-untrusted-at-generation.md)); with a public key,
  an open upload would let anyone place text in front of every other visitor's questions.
- **One public key**, baked into the console bundle and declared as the only tenant, with a
  request limit and a daily token budget ([0010](0010-tenants-isolated-by-collection.md)).
  Those limits, not the secrecy of the key, are what bound the demo's cost.
- **A hosted LLM** (Groq's free tier by default), through the existing OpenAI-compatible
  adapter ([0007](0007-local-llm-by-default-hosted-opt-in.md)). The console and the Space page
  both say that questions leave the server.
- **No failed-authentication lockout** (`AUTH_FAILURE_LIMIT` is 0). The only valid key is
  public, so the lockout protects nothing, and every visitor reaches the API from nginx's
  loopback address, so one caller sending bad keys would lock everyone else out.
- **Deployed by hand**, from a manual workflow that uploads the image's sources to the Space,
  which builds it. A public deployment changes when someone decides it should.

## Consequences

- Anyone can try the query path and the gateway from a link, at no cost to run. The worst a
  determined visitor can do is spend the shared request limit or the day's token budget, which
  makes the demo answer 429 until the window or the day turns over.
- The demo is not the air-gapped deployment the project describes. That is stated wherever a
  visitor will read it, and the production image and compose stack are unchanged.
- Observability is not part of the demo; the dashboards and traces are shown in the README.
- The free tier sleeps after a period without visits; the first visitor afterwards waits for
  the container to start (the embedding model is loaded from the image, not downloaded).
- Two settings (`CHROMA_MODE`, `DOCUMENT_WRITES_ENABLED`) now exist in the main code base for a
  deployment shape other than production. Both default to the production behaviour.

## Alternatives considered

- **A free virtual machine** (Oracle Cloud's always-free tier) running the whole compose stack,
  Ollama included. It would show everything, observability too, but it needs a payment card to
  register and a server to keep patched; a demo should not become an operations burden.
- **Hosting the console statically and the API elsewhere.** Two deployments, CORS between them,
  and the same single-container API problem on the other side.
- **Letting visitors upload documents into a collection of their own.** Isolation would hold,
  but storage on the free tier is ephemeral and shared, and anything uploaded would be served
  from a public URL under the project's name.
