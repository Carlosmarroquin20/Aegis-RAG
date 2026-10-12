# 0008 — A reflected injection fails the request instead of being cleaned up

**Status:** Accepted (recorded retrospectively)

## Context

The output guard scans every model response. When it finds the response echoing an
injection payload — text such as "ignore all previous instructions", or something shaped
like a system-prompt dump — the pipeline has evidence that a retrieved document steered the
model. Something has to be returned to the caller.

## Decision

Raise `OutputReflectionError` and let it propagate. The use case deliberately does not
catch it. The request ends as an HTTP **500** with a generic body, the event increments
`aegis_output_reflections_total`, and the `AegisOutputReflectionDetected` alert fires on a
single occurrence.

The status is 500, not 400, on purpose: the caller did nothing wrong. The pipeline failed
to produce an answer it can stand behind.

## Consequences

- The caller never receives a response the system believes was manipulated. There is no
  partially trusted answer to reason about.
- The failure is loud by construction. An operator learns that the corpus contains a
  document capable of steering the model, which is the actual problem to fix; a quietly
  cleaned answer would hide it indefinitely.
- A user gets an error for a question that was legitimate. That is a real cost, and it will
  occasionally be a false positive — a document that discusses prompt injection can make a
  model quote it.
- Getting the 500 to be a *useful* 500 took separate work: an unhandled error originally
  bypassed the middleware stack, so it carried no security headers or request ID and was
  missing from the error-rate metric. The innermost error boundary exists so this failure
  is logged, counted and correlated like any other response.

## Alternatives considered

- **Redact the matched text and return the rest.** Rejected: if the model followed one
  planted instruction, the remainder of its answer is not trustworthy either, and redaction
  implies that it is.
- **Return a canned refusal with a 200.** Rejected: it disguises a security event as a
  normal answer and removes it from every error-rate signal.
- **Log only.** The sanitizer supports it (`block_reflection=False`) for evaluation
  environments. The production wiring hard-codes blocking; it is not exposed as a setting.
