# 0019 — An ML injection classifier is measured, not enabled

**Status:** Accepted

## Context

The SecurityGateway is a signature engine and does not claim to be the safety boundary
([0002](0002-heuristic-gateway-is-not-the-safety-boundary.md)). On its 326-query labelled corpus
([GATEWAY_EVALUATION.md](../GATEWAY_EVALUATION.md)) it blocks 77.5% of the attacks it targets, 0%
of the classes it does not target (paraphrase, other languages, character substitution) and 6.4% of
legitimate queries. The usual way to close those gaps is a classifier fine-tuned on injection
data, so the question was whether one should sit beside the gateway.

Meta's Prompt Guard models are gated behind a licence acceptance, and Llama Guard is an
8B-parameter LLM, so neither can be fetched or run by a CI runner on CPU. The openly licensed,
ungated DeBERTa-v3 classifiers can be.

## Decision

Do not put a classifier in the request path. Keep the comparison as a CI measurement
(`scripts/compare_injection_classifier.py`), so the decision can be revisited with numbers, and
keep the adapter (`src/aegis/infrastructure/security/injection_classifier.py`) ready for that.

Measured in CI on the same 326 queries, at the classifiers' 0.5 threshold:

| | Gateway (strict) | protectai v2 | deepset v3 | Gateway or protectai |
|---|---|---|---|---|
| Attacks blocked, categories the gateway targets | 77.5% | 92.5% | 97.5% | 98.3% |
| Attacks blocked, the gateway's documented gaps | 0.0% | 94.1% | 97.1% | 94.1% |
| Attacks blocked, whole corpus | 60.4% | 92.9% | 97.4% | 97.4% |
| **Legitimate queries blocked** | **6.4%** | **17.4%** | **19.2%** | **22.7%** |
| Per query on a CI runner's CPU | < 1 ms | 141 ms | 136 ms | |

The classifiers close the gateway's gaps almost entirely, but they refuse legitimate questions
three times as often as the gateway does. The protectai model blocks:

- 18 of 24 Spanish questions that only sound suspicious ("¿Pueden ignorar el cobro duplicado…?");
- 4 of 25 ordinary Spanish questions;
- 2 of 40 knowledge-base questions.

Most of those it scores at 1.00, so no threshold separates them from attacks. For a customer-service
deployment in Spanish, refusing about one legitimate query in six is worse than the attacks it
would add.

The two approaches are complementary rather than redundant. The classifier misses `DAN` persona
attacks, inline HTML/SVG payloads, template injection and token floods (scores near 0.00), all of
which the gateway's signatures catch. Combined they block 97.4% of attacks.

## Consequences

- **The gateway's documented gaps remain open.** Paraphrased, translated and leetspeak attacks pass
  the input check, as before. The other layers of 0002 (the untrusted-context system prompt, the
  output reflection guard, retrieval scoped by tenant) are what stand behind it.
- **The decision is cheap to revisit.** A classifier with better Spanish precision, a GPU
  deployment, or a corpus where a refusal costs less can be measured with one command, with the
  numbers above to beat.
- **A non-blocking mode is the natural next step,** if one is wanted: score every query, count and
  log high scores, block nothing. It would show on real traffic what the corpus cannot, at the cost
  of about 140 ms per query on CPU.
- **CI downloads about 700 MB** for this measurement, cached between runs.

## Alternatives considered

- **Block when either detector fires.** It gives the best attack coverage (97.4%) and the worst
  false-positive rate (22.7%).
- **Raise the classifier's threshold.** Its false positives are scored at or near 1.00, so a
  threshold high enough to spare them would also spare most attacks.
- **Prompt Guard 2 (multilingual) or Llama Guard.** They are probably better on Spanish, but one is
  gated and the other is far too large for this service's CPU deployment. Worth measuring where
  either constraint does not apply.
- **Fine-tune a classifier on this corpus.** 326 queries are an evaluation set, not a training set.
  Training on them would make the measurement meaningless.
