# Context size: live comparison of 5 and 10 retrieved passages

Whether answers improve when retrieval passes 10 passages to the model instead of 5, or whether
the extra passages dilute them. [ADR 0028](adr/0028-multi-hop-retrieval-by-question-decomposition.md)
found that 10 passages retrieve every needed document for all 29 multi-document questions, against
25 of 29 with 5. Retrieval alone does not settle the default. A wider window only matters if the
model uses it, and it costs input tokens on every question.

**Run:** 2026-11-14. Groq, free tier. Model `openai/gpt-oss-120b`. Pipeline as shipped: strict
gateway, hybrid retrieval over `examples/atencion-cliente` with `intfloat/multilingual-e5-small`,
output sanitizer, NLI faithfulness judge. Each question was answered twice, once with `top_k=5` and
once with `top_k=10`; nothing else changed. Raw results, one line per question:
[`evals/multihop/live/context-5-vs-10.jsonl`](../evals/multihop/live/context-5-vs-10.jsonl).
Reproduce with `scripts/compare_context_size.py` (it needs a hosted provider in `.env`, so it is
not run in CI).

**Questions:** 71 of the planned 79.

- All 39 questions of the multi-hop set: 29 that need two or three documents, and 10
  single-document controls.
- 32 of a fixed sample of 40 single-document questions from the retrieval set (seed 28).

The other 8 (q-030, q-079, q-087, q-089, q-103, q-116, q-127, q-138) were not answered. The run
used up the model's free daily quota of 200,000 tokens. They are all single-document questions,
and the multi-document set is complete.

**Grading:** an independent judge from another model family, `qwen/qwen3.8-27b`, received the full
text of the documents each question needs as ground truth. It saw the two answers in a random
order, fixed per question, and graded each as *correct*, *incomplete* or *wrong*. It also said
which answer serves the customer better. Every verdict that preferred one size, and every *wrong*
at 10 passages, was then checked by hand against the corpus (below).

## Headline

| | 5 passages | 10 passages |
|---|---|---|
| Multi-document (29): judged correct | 7 (24%) | **16 (55%)** |
| Multi-document: judged wrong | 13 (45%) | 7 (24%) |
| Multi-document: abstained ("the context does not cover this") | 6 | 3 |
| Multi-document: every needed document retrieved | 25 | 29 |
| Single-document (42): judged correct | 31 (74%) | 34 (81%) |
| Single-document: judged wrong | 3 | 1 |
| Single-document: abstained | 3 | 1 |
| Faithfulness to the context (NLI, mean), multi / single | 0.359 / 0.833 | 0.458 / 0.841 |
| Input tokens per answer, mean | 960 | 1,567 (+63%) |
| Output tokens per answer, median | 176 | 197 |
| Latency, median multi / single, p95 overall | 1.42 s / 1.05 s, 2.08 s | 1.60 s / 0.93 s, 2.64 s |

Which answer serves the customer better, according to the judge:

| | 10 better | 5 better | Tie |
|---|---|---|---|
| Multi-document (29) | 14 | 1 | 14 |
| Single-document (42) | 7 | 2 | 33 |

## What the hand check found

**No answer is materially better with 5 passages.** The judge preferred 5 passages three times:

- **m-03.** "Gold member: is shipping free for a mattress sent to Escuintla?" With 5 passages the
  model says no (Q150), and with 10 it says yes. The corpus does not settle this. The loyalty
  programme grants Gold members free shipping without conditions. The shipping policy charges Q150
  for bulky items outside the capital "even if the order exceeds Q500", and says nothing about
  loyalty tiers. Neither answer flags the conflict. This is a gap in the example corpus, not
  evidence for either size.
- **q-107 and q-037.** These are omissions of detail the question did not ask for: ramps besides
  parking, and "in future purchases" after the point value. Both answers are right.

**10 passages fix real errors:**

- Five abstentions with 5 passages become answers. Three are correct (m-12, m-22, q-041) and
  two leave out a condition (m-29, q-049). At 5 passages the needed passage was missing, or
  ranked too low to make the cut.
- Three wrong rules become right ones. m-06 and m-18 applied the 7-day seasonal return window to a
  stove and to a sale item. m-20 gave the phone-support hours instead of the store's Sunday hours.
- Two answers gain a condition that changes what the customer does. m-17 adds that the warranty
  still covers a washing machine the store installed. q-058 adds the bulky-item surcharge to
  "free shipping over Q500".

The other judge preferences for 10 passages (m-09, m-13, m-14, m-15, m-19, m-24, m-27, q-009,
q-031, q-065, q-085) are added detail: a requirement, a channel, a clause. They are real, but
small.

**One new error with 10 passages:** m-07 says the return window for a mattress is 7 days (it is
15). With 5 passages the model abstained on the same question, so 10 passages turned "no
answer" into a wrong one.

**Some errors do not depend on context size.** Six questions are wrong at both sizes: m-04, m-11,
m-16, m-25, m-28 and q-101. Two of them, m-11 and m-28, share one cause. The refund policy says
refunds go back through the payment method used, then adds a sentence about cash purchases. With
both in context, the model answers "cash or a credit note" for a bank transfer and for card
instalments. The documents were retrieved; the model read the wrong sentence. m-18's answer at 10
passages is in Spanish for an English question. The large model is not covered by the
wrong-language check of ADR 0027, which applies to the small one only.

## Reading the result

- Multi-document questions are where the window matters. Correct answers go from 7 to 16 of 29,
  and wrong ones from 13 to 7. That is the gap ADR 0028 measured in retrieval (25 against 29), and
  more. Even when 5 passages hold both documents, the passage carrying the second rule is often
  not among them.
- Single-document questions are not diluted. Correct answers go from 31 to 34, abstentions from 3
  to 1, and faithfulness does not drop. On a 7-document corpus, the extra passages are mostly
  neighbouring sections of the same policies, which is context the model can use.
- The cost is input tokens: about 600 more per question (+63%). On a hosted model that is a
  proportional increase in input cost and in the per-tenant daily token budget. On the Groq free
  tier it is a matching cut in how many questions fit in the daily quota. Output tokens and
  latency barely move: +21 tokens and about +0.2 s on the median multi-document answer.

## Limits

- 71 questions on a 7-document corpus, one model, one run, `temperature` as shipped. The
  multi-document effect (7 to 16 correct) is large for a set this size; the single-document one
  (31 to 34) is within the noise.
- The judge's grades are stricter than a customer would be on detail, and it made errors. The hand
  check above covers every verdict that separated the sizes, not the 47 ties.
- On a large corpus, 10 passages are a smaller share of what is relevant, and the dilution risk
  is higher. This result supports 10 passages for this corpus and pipeline. It is not a general
  rule.
