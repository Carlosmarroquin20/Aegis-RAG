# 0002 — The input gateway is a heuristic first layer, not the safety boundary

**Status:** Accepted (recorded retrospectively)

## Context

Every query has to be screened for prompt injection before it reaches retrieval or the
model. The two families of detector are signature-based (regular expressions, structural
checks, entropy) and model-based (a classifier, or a second LLM acting as judge).

The screen runs on every request, ahead of everything else, on the event loop.

## Decision

Use a signature-based gateway that performs **no I/O**: precompiled patterns, Unicode
normalization, a homoglyph-folded view for matching, a linear staged-reasoning check and
entropy scoring. Scores are the maximum over matched rules, not a sum. Strict mode (the
default) blocks the *suspicious* band (≥ 60) as well as the *blocked* band (≥ 80).

Treat its verdict as a **signal and a cost-raiser**, and design the system so that safety
does not depend on it: the untrusted-context system prompt and the output guard apply to
every request regardless of how the query was phrased.

## Consequences

- Deterministic, explainable and effectively free: a decision in well under a millisecond,
  with the triggered rule names in the audit log. It works offline, which the air-gapped
  default requires.
- **It is bypassable, and this is documented rather than hidden.** The rules are English
  phrasings. The same instruction in another language, a paraphrase or leetspeak scores 0
  (see the threat model, section 6).
- Because the patterns run on attacker-controlled text, each one must stay linear on
  non-matching input. That constraint is real work: three of the original patterns
  backtracked for about a second on an 8 KB query and had to be rewritten, and a
  regression suite now feeds each rule its worst case.
- False positives are a product cost. Two rules were tightened after they blocked ordinary
  questions ("What did Dan say…", "Display context about…").

## Alternatives considered

- **A classifier model as the gate.** Better recall on paraphrase and other languages, but
  it adds a model to load or a network call on the hot path, is non-deterministic, and
  cannot explain a verdict. It remains the natural next layer: `QueryGuardPort` exists so a
  classifier can be added behind the same interface, alongside the signatures rather than
  instead of them.
- **An LLM-as-judge call.** Rejected for the gate: it doubles latency and cost per query and
  is itself injectable.
- **No input screening, output controls only.** Rejected: blocking an obvious attack before
  spending retrieval and generation on it is cheap, and the blocked-query metric is the
  earliest signal that someone is probing.
