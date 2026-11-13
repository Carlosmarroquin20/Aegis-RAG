# 0025 — An MCP server: one read-only tool, behind the same middleware and pipeline

**Status:** Accepted

## Context

Aegis is reachable only over its REST API. The clients that would most often want a
document knowledge base today are agents (Claude Code, Claude Desktop, Cursor, custom
agents), and they discover and call tools through the Model Context Protocol. Without an
MCP endpoint each of them needs a hand-written wrapper around `/api/v1/query`, and every
wrapper is a place where the API key, the tenant or the error semantics can be got wrong.

Two directions were possible. Aegis as an MCP **client** would let its own model call
external tools; that does not fit a system whose model only answers from retrieved
documents, and it would add tool use, the riskiest capability, to the one component that
reads untrusted text. Aegis as an MCP **server** fits: it already is a tool, with a narrow
contract (a question in, a grounded answer out).

What makes this a decision rather than a feature is that the caller changes. A REST client
is usually code a person wrote; an MCP client is a model that chooses when to call, with
what arguments, and what to do with the result. Three questions follow: how the caller is
authenticated and tied to a tenant, what the agent is allowed to do, and what text from
Aegis ends up inside another model's context.

The protocol revision current at the time of writing is 2026-07-28. Its Streamable HTTP
transport no longer has protocol-level sessions (`Mcp-Session-Id` was removed), so every
request stands alone. The official Python SDK (`mcp` 2.x) serves that revision and the
earlier ones from the same server.

## Decision

**Transport: Streamable HTTP, mounted on the existing app at `/mcp`.** Not a separate
process or port. Mounting it inside `create_app()` puts every MCP request through the same
stack as the REST routes, in the same order: request ID, access log, security headers,
CORS, API key, rate limit, error boundary, body limit. Nothing is re-implemented and nothing
can drift. The server runs **stateless** (no session store, JSON responses), which matches
the protocol revision and lets any replica answer any request
([0013](0013-one-worker-per-container.md)): no sticky sessions behind the load balancer.
Clients on the 2025 initialize handshake are served too, from the same endpoint. The SDK is an
optional extra (`aegis-rag[mcp]`), imported only when the endpoint is on, and the endpoint is
off unless `MCP_ENABLED=true`.

**Authentication: the existing API key, nothing new.** `APIKeyMiddleware` already resolves
the key to a tenant before the request reaches `/mcp`. The tool handler reads the tenant and
`client_id` from the request state, exactly as the REST routes do; neither ever comes from
a tool argument. MCP clients that support remote servers can send a static header
(`X-API-Key`), so no client is excluded. A bad key is still a 403, which matters here: MCP
clients treat a 401 as the start of an OAuth flow, and there is no authorization server to
discover. Becoming an OAuth 2.1 resource server (protected-resource metadata, tokens from an
external identity provider) is left to a later record, when there is a real IdP to trust.

**Origin validation.** The transport specification requires rejecting requests whose
`Origin` is not allowed, with 403, to stop DNS rebinding against a server bound to
`127.0.0.1` (which is how compose publishes the API). CORS does not do this: it only
decides what a browser may read, not whether the server acts. Requests to `/mcp` carrying an
`Origin` outside `CORS_ALLOWED_ORIGINS` are refused; requests without one (every non-browser
client) pass. The list is the configured one even with `DEBUG=true`, which opens CORS to every
origin for the web console: the compose stack runs in debug on loopback, exactly where
rebinding applies. The SDK's own Host/Origin check is turned off, because by default it pins
`Host` to localhost, which would refuse every deployment behind a real hostname. The API key
already makes a rebinding attack blind, so this check is a second layer, not the only one.

**One tool: `ask`.** Arguments mirror `QueryRequest` (`question`, `top_k`, same bounds). It
runs `QueryRAGUseCase.execute()` unchanged, so the security gateway, the tenant's token
budget, hybrid retrieval, the output sanitizer and the faithfulness check all apply. The
tool is annotated `readOnlyHint: true`, `destructiveHint: false`, `openWorldHint: false`,
and its name and description are constants in code: a tool description is text the client's
model reads as instructions, so it must never be built from configuration or documents.

**No write tools.** Ingest and delete stay REST-only. An agent is exactly the caller that a
poisoned document or web page can steer, and a `delete_document` tool would turn an indirect
prompt injection anywhere in that agent's context into data loss in Aegis (OWASP LLM06,
excessive agency). Writes remain an operator action through an interface a person drives.

**What the result carries.** The text content is the sanitized answer. The structured
content adds `query_hash`, `threat_level`, the faithfulness report when there is one, and,
for each source, its metadata and relevance score, **without** `content_preview`. The REST
response includes that preview (200 raw characters of the chunk); here the consumer is a
model, and raw chunk text is the one part of the response that has not passed the output
guard. Documents are untrusted at generation time
([0003](0003-documents-trusted-at-ingestion-untrusted-at-generation.md)); handing their text
unsanitized to someone else's model would make Aegis a carrier for indirect injection.

**Errors are tool results, not protocol errors.** The model should see a refusal and stop,
not retry a transport failure. Each maps to `isError: true` with a fixed message:

| Cause | REST | MCP tool result |
|---|---|---|
| Gateway block (`SecurityViolationError`) | 400 | Same vague message and truncated `query_hash`; no rule disclosed |
| Token budget used (`TokenBudgetExhaustedError`) | 429 | Message with the UTC reset time |
| Reflected injection (`OutputReflectionError`) | 500 | "Blocked by the output guard"; no partial answer ([0008](0008-output-reflection-fails-the-request.md) holds: the request fails, nothing is cleaned up and returned) |
| LLM / vector store down or slow | 503 / 504 | "Temporarily unavailable" |
| Anything else | 500 | Generic message; cause in the log, keyed by request ID |

Missing key, lockout, rate limit and body-size refusals happen in the middleware before the
MCP layer and stay HTTP statuses (403, 429, 413), which the protocol permits.

**No streaming.** Tool results are returned whole. Token streaming in MCP would mean
progress notifications carrying partial text, which re-opens the problem
[0018](0018-stream-whole-sanitized-sentences.md) solved for SSE, for a client that
generally waits for the full result anyway.

## Consequences

- Any MCP client can use a tenant's documents with one line of configuration, for example
  `claude mcp add --transport http aegis https://host/mcp --header "X-API-Key: …"`, and gets
  exactly the controls a REST caller gets: same tenant isolation, rate limit, token budget,
  metrics and audit log.
- Protocol traffic counts against the tenant's request rate limit: `initialize` and
  `tools/list` are requests too. With one tool and a stateless server that is a small
  constant per connection; the token budget, not the request count, is what bounds cost.
- The path label for every MCP call is the mount's `/mcp`, so label cardinality stays
  bounded. Per-tool visibility comes from a new counter,
  `aegis_mcp_tool_calls_total{tool, outcome}`, and a span per call; LLM usage is attributed
  to the tenant through the existing `client` label.
- The body limit for `/mcp` is far lower than the 51 MB upload envelope: a tool call carries
  at most an 8 KB question, so the endpoint caps request bodies at 64 KB.
- Agents lose the excerpt preview. An agent that wants the passages themselves to reason
  over is not served by this tool; that is the trade-off for not forwarding unsanitized text.
- One more dependency. The API image bundles it, like the Redis and OpenTelemetry extras, so
  the flag works without a rebuild; it is not imported unless the flag is on. It goes through
  `pip-audit` and Trivy like the rest.
- The public demo does not enable it: its key is public and its budget is shared, and an
  agent loop would spend the day's tokens in minutes.

## Implementation

- `src/aegis/interface/mcp/server.py` holds the tool, the Origin check and the endpoint, and is
  the only module that imports the SDK. It sits in the interface layer, so the architecture test
  needed no change.
- `build_query_use_case(collection)` in `interface/api/dependencies.py` builds the pipeline for
  REST and MCP alike.
- `create_app()` adds `/mcp` as a plain route to the endpoint's ASGI app, after the REST
  routers, and enters the SDK's task group inside the application lifespan.
- `MCP_ENABLED` (default `false`) in settings, `.env.example` and compose;
  `aegis_mcp_tool_calls_total{tool, outcome}` in the metrics module.
- Tests: `tests/unit/test_mcp_server.py` covers each error row above, the result shape, where
  the caller's identity comes from, the tool definition and the Origin check.
  `tests/integration/test_mcp_endpoint.py` drives the real app with the SDK's own client in both
  protocol eras: tenant isolation, a gateway block and a spent budget returned as tool errors,
  no document text in any result, 403 without a key or from a foreign origin (also in debug),
  413 above the body cap, and the fixed path label.

## Alternatives considered

- **A separate MCP process that calls the REST API.** Clean separation, and it could also
  serve stdio. Rejected: a second service to deploy, an extra network hop, and the API key
  has to be relayed through it, which is one more place to leak it. Clients that only speak
  stdio can use a generic stdio-to-HTTP bridge on their side.
- **stdio inside the API package.** The pipeline would run in the client's process with no
  middleware, no tenant and no rate limit: every control this project is about, skipped.
- **A `search_documents` tool returning the matching chunks.** Useful to agents that want to
  do their own synthesis, and the obvious second tool. Deferred, not rejected: before it
  exists its output needs the same treatment as an answer (markup stripped, reflection
  check), and it needs its own evaluation of what passing raw passages to an unknown model
  costs. It would be a new record.
- **Write tools (ingest, delete).** Rejected; see "No write tools".
- **Aegis as its own OAuth authorization server.** Rejected: issuing tokens, consent screens
  and client registration are an identity product, not part of a RAG API. When OAuth is
  needed, Aegis should validate tokens from an existing provider.
- **A separate port or app for MCP.** Rejected: it would need its own copy of the middleware
  stack, and the two copies would drift.
