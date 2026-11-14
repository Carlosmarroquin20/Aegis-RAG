# 0029 — Retrieve ten passages by default

**Status:** Accepted
**Date:** 2026-11-14

## Context

[ADR 0028](0028-multi-hop-retrieval-by-question-decomposition.md) measured retrieval for
questions that need two or three documents. With the default of 5 passages, every needed
document was retrieved for 25 of 29 questions, and with 10 passages for all 29. Splitting the
question into sub-queries did no better than 10 passages and cost a model call, so it was not
adopted. The candidate fix was a wider window for every question, because the router's
complexity classifier recognises only 7 of the 29 multi-document questions. ADR 0028 left that
change open: retrieval coverage does not show whether the model uses the extra passages, or
whether they dilute the answers to single-document questions, which are most of the traffic.

That was measured live in [CONTEXT_SIZE_LIVE_EVALUATION.md](../CONTEXT_SIZE_LIVE_EVALUATION.md).
`openai/gpt-oss-120b` answered 71 questions through the shipped pipeline, once with 5 passages
and once with 10. A judge from another model family (`qwen/qwen3.8-27b`) graded both answers
against the full text of the documents each question needs. Every verdict that separated the
two sizes was then checked by hand.

| | 5 passages | 10 passages |
|---|---|---|
| Multi-document (29): correct / wrong | 7 / 13 | 16 / 7 |
| Single-document (42): correct / wrong | 31 / 3 | 34 / 1 |
| Abstained, all questions | 9 | 4 |
| Input tokens per answer, mean | 960 | 1,567 |

The hand check found no answer that is materially better with 5 passages. It found one new error
with 10: a 7-day return window quoted for a mattress, where the model had abstained with 5.

## Decision

**The default `top_k` is 10.** It lives in one place, `DEFAULT_TOP_K` in
`src/aegis/application/dtos/rag_dtos.py`. It is the default of `QueryRequest.top_k` and of the
MCP tool's `top_k` argument, and the console starts at 10. The bounds stay 1 to 20, and a caller
that sends `top_k` gets what it asks for.

Every question gets 10 passages, not only the ones the classifier marks complex, since the
classifier misses most multi-document questions. The router itself is unchanged. It still
decides small or large model on the question's wording, and it now sees the wider context.

## Consequences

- Multi-document answers judged correct go from 7 to 16 of 29 on the example corpus.
  Single-document answers are not diluted: 31 correct of 42 become 34, with no drop in NLI
  faithfulness.
- About 600 more input tokens per question (+63%). Output tokens and LLM latency barely move
  (+21 tokens, about +0.2 s median on multi-document questions). On a hosted model, input cost
  rises in proportion. Each tenant's daily token budget covers fewer questions: the public demo's
  400,000 tokens a day go from roughly 350 questions to roughly 225.
- With `FAITHFULNESS_CHECK` on, the NLI judge checks every answer sentence against every premise
  in the context, so its work grows with the context. On CPU, its median time per answer rose
  from 1.75 s to 3.68 s (p90 from 3.9 s to 7.4 s, 20 recorded answers). The check is off by
  default, but a deployment that turns it on should expect that cost.
- `tests/unit/test_query_route.py` pins the default seen by the use case, and
  `tests/unit/test_mcp_server.py` checks that the MCP tool's default matches the REST one. The
  k6 load profile still sends `top_k: 5` explicitly. Its recorded measurements were taken that
  way, so they stay comparable.
- **Limits.** One model, one run, 71 questions on a 7-document corpus. On a large corpus, 10
  passages are a smaller share of what is relevant and the risk of distraction is higher.
  Revisit with `scripts/compare_context_size.py` when the corpus or the model changes.

## Alternatives considered

- **Keep 5 and let callers ask for more.** Callers do not know which of their questions span
  documents, and the console, the MCP tool and most API clients send the default.
- **10 passages only for questions the router classifies as complex.** It reaches 7 of the 29
  questions that need it (ADR 0028), and the classifier is deliberately not tuned per miss.
- **Question decomposition.** Rejected in ADR 0028: it costs a model call and retrieved less
  than 10 passages do.
- **More than 10.** 10 already retrieves every needed document for all 29 multi-document
  questions on this corpus. Beyond that, the extra passages would only add tokens.
