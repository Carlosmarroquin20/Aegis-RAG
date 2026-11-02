# 0017 — Answer faithfulness is judged by a local NLI model, sentence by sentence

**Status:** Accepted

## Context

The retrieval gate ([0012](0012-retrieval-quality-is-a-merge-gate.md)) checks that the right
document reaches the model; nothing checked what the model then says. A RAG answer can cite the
right document and still change a figure, contradict it, or add a policy that is not there. That is
the failure users notice.

The usual remedy, an LLM judging each answer, is what 0012 rejected for CI: slow, paid per run and
not deterministic, so a red build would not reliably mean a regression. It would also send every
answer and its context to a hosted model, which the air-gapped default
([0007](0007-local-llm-by-default-hosted-opt-in.md)) rules out.

## Decision

Judge faithfulness with a natural-language-inference model, locally. An answer is split into
claims (sentences). Each claim is read against premises from the retrieved context, and a premise
supports it by its entailment probability minus its contradiction probability. A claim is
supported at 0.5 or more. The score is the share of supported claims, and the unsupported sentences
are named. An answer that declines for lack of context is an abstention, not a failure. Behind
`FaithfulnessJudgePort` and `EntailmentModelPort`, the model is
`MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli`, loaded without `trust_remote_code`.

The judge is measured by `scripts/evaluate_faithfulness.py` on 80 labelled answers written against
the example corpus. 43 of them are faithful (supported, abstaining, or ending in a courtesy). 37 are
unfaithful in a named way: a wrong detail, a contradiction, an invented addition, or outside
knowledge. Measured in CI:

| Judge | Unfaithful caught | Faithful flagged | Accuracy | Per answer |
|---|---|---|---|---|
| Word overlap (reference) | 35% | 12% | 64% | — |
| MiniLM-L6, whole chunks as premises | 43% | 16% | 65% | 0.3 s |
| mDeBERTa-base, whole chunks | 73% | 21% | 76% | 25 s |
| MiniLM-L6, context sentences | 68% | 12% | 79% | 0.3–0.4 s |
| **MiniLM-L12, context sentences, minus contradiction** | **65%** | **7%** | **80%** | 0.5–0.9 s |

The first design offered whole chunks as premises and failed. NLI models are trained on premises of
a sentence or two, and a chunk let "30 días" for the general rule vouch for "30 días" for
mattresses. Following SummaC (Laban et al., TACL 2022), premises are now the context's sentences
plus each neighbouring pair. Stored chunks keep the source's hard line breaks, so lines are rejoined
first, and table rows are kept whole. Pairs are batched by length. That one change moved the small
model from 43% to 68% of unfaithful answers caught. Subtracting contradiction made no difference
for L6 and a small one for L12. It is kept because it is the published method and costs nothing.

L12 is the default over L6: it flags 1 of 38 correctly supported answers where L6 flags 3. Fewer
false alarms matter more for a signal people are asked to act on, and the cost is under a second
per answer. mDeBERTa caught more with whole chunks, but at 25 s per answer on CPU it could not be
the API's check, and it did not justify a 35-minute CI job to keep measuring.

CI fails below recall 0.62 or above a 0.10 false-flag rate, just under the measurement, on 0012's
principle: one more miss or false flag passes, two fail.

## Consequences

- **What it catches and what it misses.** It catches every invented addition (6 of 6), most
  contradictions (6 of 9) and all outside-knowledge claims (4 of 4). Wrong details are its weak
  spot: 8 of 18. A changed figure in an otherwise familiar sentence still reads as entailed, the
  known blind spot of NLI models for numbers. The report lists every miss, so this stays visible.
- **Courtesies are flagged.** A closing like "Si tiene más preguntas, con gusto le ayudo" is
  supported by nothing, so both courtesy answers in the set count as false flags. The system prompt
  asks for plain answers, which keeps them rare.
- **It is a signal, not a gate, at runtime.** With `FAITHFULNESS_CHECK` on, the answer is still
  returned, with its score and unsupported sentences. A judge failure costs the score, never the
  answer. `AegisUnsupportedAnswersHigh` fires when over a quarter of answers are flagged for 15
  minutes. At 65% recall, a quiet dashboard does not prove every answer is faithful; a loud one is
  worth looking at.
- **It is off by default.** The model is about 470 MB, baked into the image only on request, and
  adds its inference time to every query.
- **`--live` measures the pipeline itself.** It runs the configured LLM end to end and reports how
  faithful its answers are. It needs a running LLM, so it is not part of CI.
- **The evaluation set is small and synthetic.** It has 80 answers in one domain, and the 4 English
  ones are too few to read alone. Like 0012's gate, it catches regressions; it is not a benchmark.

## Alternatives considered

- **An LLM as judge.** It probably catches more, wrong figures included, but it is non-deterministic
  for a CI gate, paid per run, and hosted. 0012's reasons stand.
- **Whole chunks as premises.** Measured above: 43% caught.
- **A larger NLI model (mDeBERTa).** Better with whole chunks, but too slow on CPU for the API.
  Worth re-measuring with sentence premises on a GPU deployment.
- **A rule that every figure in a claim must appear in the context.** Cheap and deterministic, but
  in this corpus the wrong figures usually do appear, attached to another case ("30 días" exists,
  for general returns). It would catch little of what NLI misses.
