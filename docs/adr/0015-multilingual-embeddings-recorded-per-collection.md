# 0015 — A multilingual embedding model, recorded in every collection

**Status:** Accepted

## Context

The service answers Spanish questions over Spanish documents as well as English ones, and the
default embedding model, `all-MiniLM-L6-v2`, was trained on English. Measured on the example
corpus, it found the right document in the top five for 88% of Spanish questions and 50% of the
English ones asked about Spanish documents.

Changing the model raised a second problem. ChromaDB records the *type* of embedding function a
collection was built with, not the model: switching to another SentenceTransformers model of the
same dimension is accepted in silence, and every query is then compared with vectors from an
incompatible space. Answers come back ranked, plausible and wrong, with no error anywhere.

## Decision

Make `paraphrase-multilingual-MiniLM-L12-v2` the default, baked into the image so the air-gapped
deployment needs no download. Each new collection records the model that built it in its
metadata (`aegis:embedding_model`). At startup the API refuses to serve a collection recorded
with a different model and says how to recover. A collection created before the record existed
is served with a warning.

## Consequences

- On the same 38 questions: Spanish hit@1 from 62% to 79% and hit@5 from 88% to 97%; English
  hit@5 from 50% to 100%; MRR from 0.708 to 0.863. The retrieval gate
  ([0012](0012-retrieval-quality-is-a-merge-gate.md)) now holds these figures.
- About 40 ms more per query on CPU (60 ms against 21 ms) and about 390 MB more in the image.
- Changing the model means re-ingesting into a new collection. The guard makes that explicit,
  where before it would have been a silent loss of quality.
- With several tenants ([0010](0010-tenants-isolated-by-collection.md)), every tenant's
  collection is checked at startup, so one built with another model stops the deployment rather
  than failing that tenant's first query.

## Alternatives considered

- **Keep the English model.** Rejected on the measurement: a Spanish-speaking deployment would
  miss one question in eight before the model was even involved.
- **A larger multilingual model.** Better scores are likely, at several times the size and the
  latency on CPU; the small model already clears the gate.
- **Trust the operator not to change the model.** That is what happened before, and the failure
  is invisible: nothing errors, the answers just get worse.
