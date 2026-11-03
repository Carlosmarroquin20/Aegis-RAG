# 0018 — Stream answers as whole, sanitized sentences

**Status:** Accepted

## Context

On CPU, a local model takes tens of seconds to answer, and the API returned nothing until the
last word. Streaming the answer as it is generated is the standard fix, but it conflicts with
two decisions this service rests on:

- The output guard checks the *whole* answer before anything is returned: it strips markup,
  flattens Markdown, and fails the request if the answer echoes an injection payload
  ([0008](0008-output-reflection-fails-the-request.md)). Streaming sends text before the whole
  answer exists, and sent text cannot be taken back.
- Failures map to status codes: 400 for a blocked query, 429 for a spent budget, 503 or 504 for
  a backend that failed. A streamed response commits to 200 when its first byte goes out.

## Decision

Add `POST /api/v1/query/stream`, which answers with server-sent events: `start` (the sources),
`delta` (text to append), then `end` (the complete answer). `POST /api/v1/query` is unchanged.

**Text leaves only as whole sentences, already sanitized.** A `StreamingOutputGuard` sits between
the model and the client:

- It releases text only up to a sentence boundary, never mid-sentence.
- It releases a boundary only once 80 more characters have arrived after it. The reflection
  check runs over the whole buffer, that margin included, so a payload straddling a boundary is
  caught before the text in front of it goes.
- It never uses a boundary inside an unclosed HTML tag or Markdown link.
- What it releases is the sanitizer's own output for that prefix, so markup is stripped and
  links rewritten exactly as in a buffered answer.

**Everything that can refuse runs before the first byte.** The security gateway, the token
budget, retrieval, and generation up to the first released sentence all run before the response
starts. A blocked query is still a 400, a spent budget a 429, and a backend that fails before
any text a 503 or 504. A reflection in the first sentence is still the 500 of 0008. The LLM
fallback ([0009](0009-llm-fallback-behind-circuit-breakers.md)) still switches backends, but
only before the first piece of text; after that, the answer belongs to the backend that started
it.

**After the first text, a failure is an `error` event.** A reflection detected later, or a model
that fails mid-answer, ends the stream with `{"code": ..., "message": ...}` and the client must
discard what it showed. The reflected text itself is never sent. The `end` event carries the
complete sanitized answer, and the client replaces the assembled deltas with it. That covers the
rare case where sanitizing more text would rewrite text already sent: the guard stops sending
deltas rather than send a correction.

## Consequences

- **Time to first text becomes the first sentence plus 80 characters,** instead of the whole
  answer. `aegis_stream_first_delta_seconds` measures it, and `aegis_streams_total` counts how
  streams end: completed, blocked, failed or disconnected.
- **0008 holds only partly for streams.** A user may see the start of an answer that is then
  withdrawn. The text the guard objects to never reaches them, but earlier sentences of a
  manipulated answer can. Clients must honour `error`; the bundled console clears the answer.
  Callers that cannot accept this keep using `/query`.
- **Closing the connection stops the model.** The LLM request is closed when the client goes
  away, so an abandoned answer does not keep generating. Its tokens are then not charged to the
  tenant's budget, because the usage arrives only with the last chunk; abandoned answers cost
  the deployment, not the tenant.
- **Proxies must not buffer the response.** The response sets `X-Accel-Buffering: no`, which nginx
  honours. Other proxies need their buffering off for this path.
- **The adapters speak each provider's wire format:** Ollama's NDJSON, and OpenAI-compatible
  server-sent events with `stream_options.include_usage`, so streamed answers are still billed.
  A backend without streaming falls back to one delta carrying the whole answer, through the
  port's default.

## Alternatives considered

- **Stream raw tokens and sanitize at the end.** Fastest first byte, and it abandons LLM05 for
  streamed answers: markup, hidden link targets and reflected payloads would reach the client
  before any check.
- **Buffer the whole answer and stream it out afterwards.** Keeps 0008 intact, and gives the
  user nothing sooner, which is the only point of streaming.
- **Release token by token, re-checking the prefix each time.** A reflection phrase would then be
  half-sent before its last word arrived and matched. Whole sentences plus a margin are what make
  "never sent" true.
- **Start the 200 at once and report every failure as an event.** It would hide blocked queries,
  spent budgets and outages from status-code dashboards and alerts, the same objection 0008
  raises against a 200 with a canned refusal.
