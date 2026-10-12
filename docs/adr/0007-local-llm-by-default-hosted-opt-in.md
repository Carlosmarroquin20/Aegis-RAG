# 0007 — Local LLM by default; hosted providers are opt-in through one adapter

**Status:** Accepted

## Context

The system was built to run entirely on local infrastructure, with Ollama as the only LLM
backend. That keeps every query and document on the host, and it is slow: on CPU a single
answer took about 90 seconds in testing. A hosted model answers in about one. Supporting
one changes a core property of the system — data leaves the host — so it is an architectural
decision, not a configuration detail.

## Decision

Keep the local backend as the default and add hosted providers as an explicit opt-in
(`LLM_PROVIDER=openai_compatible`), implemented as **one** adapter for the OpenAI Chat
Completions wire format rather than one adapter per vendor.

Guard rails around the opt-in:

- Startup logs a warning naming the provider host whenever it is enabled; the system cannot
  run in that mode silently.
- Settings validation refuses to start without a base URL, model and key, or with a
  non-HTTPS URL outside debug mode.
- The API key is a `SecretStr`, sent only as a header. Error text is built from the status
  code and the provider's error *code*; the provider's free-text message is never used,
  because providers echo request details there and the text reaches the logs.
- The system prompt and context framing live in one shared module, so a hosted model
  receives exactly the injection defence the local one does.

## Consequences

- One setting moves between an air-gapped deployment and a fast one, with no code change.
  Any provider that speaks the format works — Groq, OpenRouter, Mistral, a self-hosted vLLM
  gateway — by changing three values.
- With a hosted backend the query and the retrieved excerpts are sent to a third party. No
  technical control changes that; it is a data-handling decision for the operator, and the
  documentation says so in those words.
- Provider quotas become a failure mode. A rate limit or a rejected key surfaces as a 503
  through the existing error mapping, and the readiness probe caches its result so
  unauthenticated `/ready` calls cannot spend quota.

## Alternatives considered

- **Hosted by default.** Rejected: it would silently break the property the project is
  built around, and a secure-by-default system should require a decision to send data out.
- **A dedicated adapter per provider.** Rejected as duplication; the providers of interest
  share a wire format, and the differences that matter are a URL and a model name.
- **A third-party abstraction library.** Rejected: it adds a large dependency to a
  security-sensitive path in order to wrap one HTTP call, and takes the error handling and
  credential hygiene out of this codebase's control.
