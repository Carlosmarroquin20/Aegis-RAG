# 0022 — The faithfulness judge drops reply words and checks figures against the context

**Status:** Accepted

## Context

On the larger evaluation set ([0021](0021-larger-evaluation-sets.md)) the NLI judge
([0017](0017-answer-faithfulness-judged-by-nli.md)) caught 54.6% of unfaithful answers and flagged
12.5% of faithful ones. The errors had two clear causes.

- **Reply words.** 8 of the 12 correct answers it flagged began with a bare "No," or "Yes,". "No,
  the stores are closed on 25 December" scored 0.00. The judge never sees the question, so the NLI
  model reads that word as a negation of the sentence after it.
- **Invented figures.** It caught 19 of 51 wrong figures. NLI models read "60 days" as entailed by
  "30 days" in an otherwise identical sentence. Of the 32 it missed, 16 contained a number that
  appears nowhere in the source document, and no faithful answer contained one. 0017 had rejected a
  figure check because, on 18 examples, the wrong figures usually did appear in the context. The
  larger set showed that was not true in general.

## Decision

Add two corrections to the judge, both deterministic and free of models.

- **A leading reply word is dropped before a claim is judged.** This covers "Sí", "Si", "No" and
  "Yes", only when punctuation follows, and only when at least three words remain. "No hay entregas
  los domingos" keeps its negation. The sentence is still reported as written.
- **A claim with a figure the context never states is unsupported,** whatever the NLI model says.
  The figures are named in `SentenceSupport.ungrounded_figures`. Only figures written in digits are
  checked. The same value written another way is matched:
  - thousands separators: a separator followed by exactly three digits groups thousands in either
    convention, so "Q1.000" is "Q1,000";
  - decimal commas: "0,10" is "0.10";
  - times: "20:00" also grounds "20", "8:00" and "8";
  - numbers the context spells out: "dos referencias" grounds "2".

  A figure in the user's question is not an invention either. `FaithfulnessJudgePort.assess` now
  takes the question, and both the normal and the streaming pipeline pass it.

`scripts/evaluate_faithfulness.py --keep-reply-words` and `--no-figure-check` switch each
correction off, so each can be re-measured on its own.

## Measurements

Measured in CI on the 271 labelled answers:

| Judge | Unfaithful caught | Faithful flagged | Accuracy |
|---|---|---|---|
| Before (0021) | 54.6% (65/119) | 12.5% (19/152) | 73.1% |
| Reply words dropped | 55.5% (66/119) | 7.9% (12/152) | 76.0% |
| Figure check | 67.2% (80/119) | 12.5% (19/152) | 78.6% |
| **Both** | **67.2%** (80/119) | **7.9%** (12/152) | **81.2%** |
| Both, English / Spanish | 65.6% / 69.0% | 9.3% / 6.5% | 79.4% / 83.0% |

Each correction does what it was meant to do, and they do not interfere. Dropping reply words
removes 7 of the 12 false flags on correctly supported answers. The figure check raises wrong
figures caught from 19 to 33 of 51 and flags no faithful answer. English, the weaker language
before, gains the most: recall from 47.5% to 65.6%. The judge takes no longer per answer (650 ms,
against 780 ms before, within run-to-run noise).

CI now fails below recall 0.66 or above a 0.09 false-flag rate, on 0012's principle: one more miss
or false flag passes, two fail.

## Consequences

- **Some answers were caught for the wrong reason, and no longer are.** "No, la instalación de
  lavadoras es gratuita en todo el país" contradicts the context. Before, it was flagged only
  because of its "No,". Without that word, the rest reads as entailed. One outside-knowledge claim
  ("Sí, se acepta American Express...") is lost the same way. Two contradictions are gained, so
  recall is unchanged. A judge that is right for the right reasons is worth more than one that is
  right by accident.
- **A figure from the wrong case still passes.** "30 días" for mattresses is grounded by the "30
  días" for general returns. The figure check cannot see which case a number belongs to, and NLI
  still misses most of these: 18 wrong figures remain uncaught.
- **The remaining false flags are mostly courtesies.** All 7 answers ending in a courtesy are still
  flagged, and 5 supported answers that combine several facts or paraphrase a table row are flagged
  too. The system prompt asks for plain answers, which keeps courtesies rare.
- **Contradictions are now the weakest kind**, at 19 of 35 caught. Most reuse the context's own
  wording and figures and change only the policy ("se acepta" for "no se acepta"). That is the
  known limit of a small NLI model. A larger model on a GPU, or an LLM judge where one is
  acceptable, is the remaining remedy.
- **The check can call a correct answer unfaithful** when the model computes a figure from the
  context ("Q450 left to free shipping") or converts one ("about Q1,000"). The answer is still
  returned, and the sentence is named with the figure that caused it. That makes the flag easy to
  review, and it is the right default for a monitoring signal.

## Alternatives considered

- **Give the NLI model the question as well.** "Question + answer" as the hypothesis makes it longer
  than the premises the model was trained on, and a question is not a claim. Dropping only the reply
  word fixes the measured cause without changing what is judged.
- **Check number words in claims too ("five years").** It would catch one more answer in this set,
  but it would have to tell "one of the products" from "1 product" in two languages. Only the
  context's number words are read, so that grounding is generous and flagging stays strict.
- **Treat any figure mismatch as a contradiction score instead of a veto.** Blending it into the NLI
  score would need a weight tuned on this one corpus. A veto that names the figure is simpler to
  explain and to review.
