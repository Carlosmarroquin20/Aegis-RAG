# 0012 — Retrieval quality is a merge gate

**Status:** Accepted

## Context

Retrieval decides whether the model ever sees the answer: a question whose document is not
among the retrieved chunks is answered "not enough information", however good the model is.
It depends on the parser, the chunker and the embedding model, and a change to any of them can
make it worse while every unit test still passes, because the tests check behaviour, not
ranking. `scripts/evaluate_retrieval.py` already measured it (hit@k and MRR on 38 labelled
questions over the example corpus), but only when someone remembered to run it.

## Decision

Run the evaluation in CI as its own job, against the default model, and fail below
**hit@5 0.94** and **MRR 0.82**. The measured values are 0.974 (37 of 38 questions) and 0.863.

The thresholds sit just under the measurement on purpose: one more miss still passes, two fail,
and so does a ranking that drifts down without losing a hit. The questions were written before
the models were compared and the corpus is the one the README tells people to try, so the gate
measures something a user would notice.

## Consequences

- A regression in parsing, chunking or the default model fails the build instead of reaching
  users as unanswered questions.
- The job needs the embedding model, about 470 MB, downloaded once and cached between runs; the
  first run after a model change pays for the download.
- The numbers are deterministic: the same model files give identical figures in CI and on a
  laptop, so a failure is a real change, not noise.
- 38 questions are a small sample, and one domain. The gate catches regressions; it is not a
  benchmark, and the thresholds are raised when retrieval improves, never lowered quietly.
- The job is not yet one of the checks the branch ruleset requires; adding it there is what
  makes it binding for pull requests from others.

## Alternatives considered

- **Keep the evaluation manual.** Rejected: it was already written and already accurate, and
  went unrun.
- **Judge end-to-end answers with an LLM.** It would measure more, but it is slow, costs money
  per run, and is not deterministic, so a red build would not reliably mean a regression.
- **Require the exact current figures.** Too brittle: an improvement that moves one question
  down a rank while moving two up would fail.
