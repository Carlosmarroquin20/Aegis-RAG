# 0024 — A public demo: the gateway in the browser, the full stack in one container

**Status:** Accepted (revised the same day: Docker Spaces turned out to need a paid plan)

## Context

The project is meant to be looked at, and almost nobody who looks at it will clone the
repository and start eight containers. A hosted demo removes that step, with two constraints:
it must cost nothing, and the security properties it demonstrates must survive being open to
anyone on the internet.

The first plan was one container on Hugging Face Spaces' free CPU tier. The image was built
and verified, but creating the Space showed that Docker (and Gradio) Spaces now require a paid
plan; only static Spaces are free. The free hosts that do run containers (Render, Koyeb) give
512 MB, less than the embedding model and its runtime need, and the rest ask for a payment
card. Free and without a card leaves static hosting.

## Decision

Two demos, for two audiences.

**The gateway playground** (`deploy/gateway-playground/`), free and public, on the GitHub Pages
site the repository already has. A static page loads Pyodide and runs the SecurityGateway in
the visitor's browser.

- It runs the real module, not a port: the build follows the gateway's imports from `src/`
  and copies exactly those files, and refuses to build if the gateway starts importing a
  package the browser cannot load. `structlog` is replaced by a no-op stand-in; pydantic comes
  with Pyodide.
- A unit test imports the built tree on its own, so a file missing from the page fails the
  build here, not in a visitor's browser. A workflow republishes the page whenever the gateway
  changes on main.
- Nothing typed leaves the tab, so there is no key, no quota and nothing to abuse. The page
  says what it is not: the first layer only, with two answers recorded from the full stack to
  show what follows.

**The single-container image** (`deploy/huggingface/`), for anyone with a Docker host (a paid
Space, Cloud Run, a VM). It keeps the shape it was built with:

- nginx serves the console and forwards the API paths to uvicorn on loopback; Chroma runs in
  the API process (`CHROMA_MODE=embedded`). `/metrics` is not forwarded.
- The corpus is indexed at build time, through the same ingestion pipeline as an upload.
- Read-only: `DOCUMENT_WRITES_ENABLED` is off, and nginx refuses write methods on the documents
  route as well. Ingested documents skip the gateway
  ([0003](0003-documents-trusted-at-ingestion-untrusted-at-generation.md)); with a public key,
  an open upload would let anyone place text in front of every other visitor's questions.
- One public key, declared as the only tenant, with a request limit and a daily token budget
  ([0010](0010-tenants-isolated-by-collection.md)): the limits, not the key's secrecy, bound
  the cost.
- A hosted LLM through the OpenAI-compatible adapter
  ([0007](0007-local-llm-by-default-hosted-opt-in.md)); the console says questions leave the
  server.
- No failed-authentication lockout (`AUTH_FAILURE_LIMIT` is 0): the only key is public, and
  every visitor reaches the API from nginx's loopback address, so one caller sending bad keys
  would lock everyone else out.
- Deployed by hand, from a manual workflow.

## Consequences

- Anyone can try the gateway from a link, on any device, at no cost to anyone, and see each
  verdict with its score, rules and timing. Evasions (zero-width characters, look-alike
  letters, missing accents) can be tried directly against the code that handles them.
- The free demo does not show retrieval, generation or the output sanitizer live. The recorded
  answers and the full image fill that gap for those who want it.
- The full image stays verified and ready, but publishing it costs money on every host
  checked, so it is not published by default.
- Two settings (`CHROMA_MODE`, `DOCUMENT_WRITES_ENABLED`) exist in the main code base for a
  deployment shape other than production. Both default to the production behaviour.

## Alternatives considered

- **Paying for a Docker Space.** The simplest way to publish the full image, and still open
  through the manual workflow; rejected as the default because the demo was to cost nothing.
- **Cloud Run or another free tier with a card on file.** Within the free allowance with one
  instance and scale to zero, but nothing caps the bill, and a public endpoint is exactly what
  attracts the traffic that would exceed it.
- **A free virtual machine** (Oracle Cloud's always-free tier) running the whole compose stack.
  It needs a payment card to register and a server to keep patched.
- **Porting the gateway's rules to JavaScript.** No runtime to download, but it would be a
  second implementation that drifts from the first, which is the opposite of a demo.
