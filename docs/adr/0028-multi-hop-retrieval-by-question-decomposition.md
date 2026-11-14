# 0028 — Question decomposition for multi-document retrieval: measured, not adopted

**Status:** Accepted
**Date:** 2026-11-14

## Context

Some questions need more than one document. "I'm a Gold member in Quetzaltenango. Do I pay for
installation?" depends on the loyalty programme (Gold members get free installation) and on the
shipping policy (installation outside the capital costs Q200). When retrieval brings only one of
the two, the model answers from half the rules, and nothing downstream notices: the answer is
faithful to the context it was given.

The usual fix is an agentic step before retrieval: a model splits the question into sub-queries,
each is retrieved on its own, and one answer is generated from the combined context. It costs an
extra LLM call per question, and it is only worth that if it retrieves what single-shot retrieval
misses. The simple alternative costs no call at all: retrieve more passages.

## Decision

**Measure first, with a go/no-go gate.** `evals/multihop/dataset.jsonl` holds 29 questions that
need two or three documents of the example corpus, and 10 single-document controls, in Spanish
and English. Each is labelled with the documents its answer depends on. The labels were read off
the corpus, and the questions were written so that the documents are needed rather than merely
related. `scripts/evaluate_multihop.py` compares, at budgets of 5 and 10 passages, retrieving the
question as one query against retrieving each sub-query of a recorded plan. For the plan, the
results of each query are interleaved without duplicates up to the budget. The planner was
`openai/gpt-oss-120b` with a strict JSON contract: 1 to 3 queries of at most 200 characters,
otherwise the question itself. The gate was set before the run: adopt decomposition only if it
retrieves every needed document more often than 10 passages do, with the same or less context.

| Strategy | Multi-document: all needed retrieved | Controls | Context |
|---|---|---|---|
| One query, 5 passages (today) | 86.2% (25/29) | 10/10 | 2,223 chars |
| Decomposition, 5 passages | 89.7% (26/29) | 10/10 | 2,263 chars |
| One query, 10 passages | **100% (29/29)** | 10/10 | 4,505 chars |
| Decomposition, 10 passages | 96.6% (28/29) | 10/10 | 4,554 chars |

**Decomposition is not adopted.** It failed the gate. At equal context it gains one question in
29, and at 10 passages it loses one that plain retrieval finds. Its plans were well formed: 39 of
39 valid, 2 or 3 queries for 26 of the 29 multi-document questions, 1 for the other 3 and for
every control. So the planner is not the problem; splitting is. A rewrite can drop the words
retrieval keys on: the question about "apartar" ranks the FAQ first, the sub-query "política de
apartado" ranks it sixth (m-22). And interleaving gives each sub-query about half the budget, so
a document that the whole question ranks eighth can fall out of ten passages (m-12, an English
question over the Spanish payments document). Plain retrieval with a wider window already
reaches every needed document on this corpus.

**The candidate fix is wider context, and it is not switched on by this record.** The router's
complexity classifier marks only 7 of the 29 multi-document questions as complex: most combine
two policies without a comparison, a second question mark or a conjunction it looks for. So
"10 passages for complex questions" would miss most of them, and the 100% figure applies only if
every question gets 10 passages. That doubles the context on every question, about 570 more
input tokens. Whether it improves answers, or dilutes them for the single-document majority, is a
question about answers, not retrieval. It needs the same kind of live comparison as ADR 0026
before the default changes.

## Consequences

- No agent in the query path, and no planner call to pay for, secure or monitor. Decomposition
  would have added a model-written query between the gateway and retrieval. That is a smaller
  injection surface than a tool-calling loop, since the planner sees only the user's question,
  but it is a surface all the same.
- The evaluation stays. `docs/MULTIHOP_EVALUATION.md` is generated from the recorded plans, CI
  checks that it is current in the Retrieval Quality Gate job without calling a provider, and a
  retrieval change that loses multi-document coverage shows up there.
- The prompt and the parser are kept in the script, so revisiting decomposition with a larger or
  messier corpus, or with a different planner, is one `--plan` run away.
- **Limits.** 29 questions on a 7-document corpus written by the same author as the labels and
  the planner prompt. The gap between the strategies is a handful of questions, inside the noise
  of a set this size. That is enough to reject a component that costs a model call; it is not
  enough to say decomposition never helps. On a large corpus, where 10 passages cover a smaller
  share of what is relevant, the balance can change.

## Alternatives considered

- **Decomposition with the original question as one more query.** It would keep the user's own
  words in play, which is what lost the misses above. It is still an extra model call to match
  what 10 passages do without one. Left for a corpus where 10 passages are not enough.
- **An iterative tool-calling agent** (search, read, decide the next search). Each step reads
  retrieved text, so the documents themselves can steer the next query: indirect injection with
  a search tool in hand. It also costs 2 to 4 model calls per question and is hard to evaluate
  deterministically. Not built, because the simpler planner already failed to beat the baseline.
- **Wider context only for complex questions.** It reaches 7 of the 29 questions that need it;
  see above.
