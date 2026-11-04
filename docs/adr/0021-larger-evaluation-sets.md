# 0021 — Larger evaluation sets, and the gates re-set on them

**Status:** Accepted

## Context

The retrieval gate ([0012](0012-retrieval-quality-is-a-merge-gate.md),
[0016](0016-hybrid-retrieval-by-default-reranker-opt-in.md)) measured 38 questions, 4 of them in
English. The faithfulness gate ([0017](0017-answer-faithfulness-judged-by-nli.md)) measured 80
answers, 9 in English, with 2 to 6 examples of most kinds of error. Both ADRs said the sets were too
small to read closely. 0016 deferred the reranker decision "with a larger evaluation set". 0017
rejected a check on figures because "the wrong figures usually do appear" in the context, which was
true of 18 examples.

At those sizes one question moved hit@5 by 2.6 points. The figures were too coarse to separate two
modes, or to give English its own number.

## Decision

Grow both sets and re-set both gates on the new measurements. The method, the models and the
defaults are unchanged.

- **Retrieval: 141 questions** over the same seven documents, 20 per document (21 for returns), 56
  in English. Each question now has a `kind`: `direct` uses the document's own terms, `paraphrase`
  asks the same thing in other words, and `scenario` describes a situation the policy has to be
  inferred from. `scripts/evaluate_retrieval.py` reports each kind separately.
- **Faithfulness: 271 labelled answers** over 146 questions, 136 in English. Every kind of answer
  appears in both languages: 133 supported, 7 ending in a courtesy, 12 abstentions, and 119
  unfaithful ones (51 wrong figures, 35 contradictions, 17 invented additions, 16 claims from
  outside knowledge).
- **Unit tests now check both sets:** unique ids and questions, known kinds and languages, at least
  15 questions per document, enough answers per language to read alone, and every abstention
  recognised as one by the judge.

The 80 original answers and 38 original questions are kept unchanged, so the old figures are
still a subset of the new ones.

## Measurements

Measured in CI on the same runner class as before.

**Retrieval** (the model sees the top 5 chunks, so hit@5 is what decides an answer):

| Mode | hit@1 | hit@5 | MRR | per query |
|---|---|---|---|---|
| dense | 70.9% | 95.7% | 0.813 | 16 ms |
| **hybrid** *(default)* | 75.9% | 96.5% | **0.850** | 16 ms |
| dense+rerank | 83.7% | 97.2% | 0.899 | 824 ms |
| hybrid+rerank | 84.4% | 97.9% | 0.905 | 813 ms |
| BM25 alone | 53.2% | 61.0% | 0.568 | < 1 ms |

| Hybrid, by slice | hit@5 | MRR |
|---|---|---|
| Spanish (85) | 96.5% | 0.870 |
| English (56) | 96.4% | 0.819 |
| direct (78) | 98.7% | 0.912 |
| paraphrase (31) | 90.3% | 0.747 |
| scenario (32) | 96.9% | 0.797 |

**Faithfulness**:

| | Unfaithful caught | Faithful flagged |
|---|---|---|
| All 271 | **54.6%** (65/119) | **12.5%** (19/152) |
| The original 80 (unchanged) | 64.9% (24/37) | 7.0% (3/43) |
| The 191 new ones | 50.0% (41/82) | 14.7% (16/109) |
| English / Spanish | 47.5% / 62.1% | 13.3% / 11.7% |

By kind: invented additions 15 of 17 caught, outside knowledge 14 of 16, contradictions 17 of 35,
wrong figures 19 of 51. Supported answers were flagged 12 times out of 133, courtesies 7 of 7,
abstentions 0 of 12.

**The gates** move to just under these figures, on 0012's principle that one more miss or false
flag passes and two fail. Retrieval: hit@5 0.955 and MRR 0.83 (from 0.97 and 0.87). Faithfulness:
recall 0.535 and false flags 0.135 (from 0.62 and 0.10). The code did not get worse: the earlier
thresholds were set on samples too small to show how good it was.

## Consequences

- **The faithfulness judge is weaker than 80 answers suggested.** It catches about half of the
  unfaithful answers, not two thirds, and it is weaker in English than in Spanish. The README and
  0017's figures are superseded by the ones above.
- **Most of its false flags have one cause.** 8 of the 12 supported answers it flags begin with a
  bare "No," or "Yes," ("No, the stores are closed on 25 December" scores 0.00). The judge does not
  see the question, so that word reads as a claim that negates the rest of the sentence. Dropping a
  leading yes or no before judging should remove most of these. That is a change to the judge, so
  it needs its own measured change.
- **0017's case against a figure check no longer holds.** 16 of the 32 wrong figures the judge
  misses contain a number that appears nowhere in the source document (60 days, 90 days, 72 hours,
  25%), and no faithful answer in the set contains one. A rule that every figure in a claim must
  appear in the context would catch those 16 without a single new false flag here. That would put
  recall near 68%. It would not catch a figure taken from the wrong case ("30 días", which does
  exist for general returns), which is what NLI misses too. Like the yes/no fix, it is a follow-up
  to measure, not part of this change.
- **Hybrid helps Spanish and does nothing for English.** BM25 finds nothing for an English question
  over a Spanish corpus (hit@5 12.5%), so for English questions fusion mostly adds noise: MRR 0.819
  against 0.824 for dense alone. On Spanish it raises MRR from 0.805 to 0.870. Hybrid stays the
  default, but this is a known limit for corpora in a language the questions are not.
- **The reranker stays opt-in.** It adds 0.055 MRR, up from 0.017 on 38 questions, and puts the
  right document first for 12 more questions. But hit@5, the context the model actually receives,
  differs by 2 questions out of 141, at about 800 ms per query on CPU. 0016's conclusion holds on
  better evidence. On a GPU, or where answers depend on the first chunk alone, it is worth turning
  on.
- **Paraphrases are where retrieval fails.** Paraphrases have the lowest hit@5 (90.3%) and MRR
  (0.747) under hybrid, and 3 of the 5 questions it ranks past fifth are paraphrases. They are the
  first place to look when changing chunking or the embedding model.
- **The sets are still synthetic and one domain.** They were written by hand against seven short
  documents. They catch regressions more reliably than before and give English a real number. They
  are still not a benchmark.

## Alternatives considered

- **Generate questions and answers with an LLM.** It would be faster and larger, but it is biased
  toward the model's own phrasing. Every label would still need checking by hand against the
  documents, which is where the time goes.
- **Keep the gates where they were and fix the judge first.** That would hold the larger set
  hostage to two changes that have not been measured yet. Re-setting the gates on what the code
  does today keeps CI honest. Each fix then raises them on its own evidence.
- **Report only the new answers.** Keeping the original 80 inside the 271 keeps the old figures
  reproducible and shows that the drop comes from the data, not from the code.
