# 0016 — Hybrid retrieval by default; the reranker is opt-in

**Status:** Accepted

## Context

Retrieval was dense similarity alone. Embeddings match meaning but blur exact terms: a product
name, a figure or a rarely seen word can rank below a chunk that is merely about the same topic.
On the example corpus, dense retrieval put the right document first for 79% of questions and in
the top five for 97%, with MRR 0.863; the one miss ("¿Cortan madera a la medida?") ranked eighth.

Two standard remedies exist. A **lexical index** (BM25) matches exact terms and misses
paraphrases and other languages, the opposite failure of an embedding, so the two can be fused.
A **cross-encoder reranker** reads the query together with each candidate, which a bi-encoder
never does, and is more accurate but far slower. Either can make retrieval better, and the
reranker can also make every query noticeably slower.

ChromaDB scores no keyword search (`$contains` filters, it does not rank), so BM25 has to come
from somewhere else.

## Decision

Put retrieval behind a `RetrieverPort` with three optional stages, chosen by `RETRIEVAL_MODE`:
`dense`, `hybrid` (dense fused with BM25 by reciprocal rank fusion), `dense+rerank` and
`hybrid+rerank`. The default is **`hybrid`**. The reranker
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, multilingual, loaded without
`trust_remote_code`) is available but off.

The decision rests on the evaluation, which now retrieves through the application's own
`Retriever`, BM25 index and reranker (`scripts/evaluate_retrieval.py --compare`). On the same 38
questions, on a CI runner's CPU:

| Mode | hit@1 | hit@3 | hit@5 | MRR | per query |
|---|---|---|---|---|---|
| dense | 78.9% | 92.1% | 97.4% | 0.863 | 18 ms |
| **hybrid** | 81.6% | 100% | 100% | **0.908** | 21 ms |
| hybrid+rerank | 86.8% | 100% | 100% | 0.925 | 891 ms |
| BM25 alone | 81.6% | 86.8% | 86.8% | 0.842 | < 1 ms |

Hybrid recovers every question dense retrieval ranked past the top three for 3 ms. The reranker
adds 0.017 MRR, two more questions answered from the first chunk, for close to a second per
query. That is small next to a local LLM's tens of seconds, but it doubles the latency of a
hosted one. On 38 questions two is within noise, so it is not worth that latency by default.

The BM25 index lives in process memory, one per collection, built from the vector store, which
stays the only source of truth. Before each search the store's chunk count (a new
`VectorStorePort.count()`) is compared with the count the index was built from, so an upload or
a delete through any replica is seen on the next query. A change that keeps the count equal is
caught by `LEXICAL_INDEX_MAX_AGE_SECONDS`. Text is folded to unaccented lower case, stripped of
a short Spanish and English stop-word list and of plurals; plural stripping alone moved BM25's
MRR from 0.782 to 0.842.

The retrieval gate ([0012](0012-retrieval-quality-is-a-merge-gate.md)) now measures the
configured mode, and its thresholds rise to hit@5 0.97 and MRR 0.87, just under the hybrid
figures on the same principle as before: one more miss passes, two fail.

## Consequences

- Better retrieval by default at no model cost: no new dependency, no download, no change to the
  image's size.
- BM25 holds every chunk of each collection in memory, and a rebuild re-reads the collection.
  Fine for thousands of chunks per tenant; a collection of millions needs a store with native
  full-text search (pgvector with `tsvector`, OpenSearch, Qdrant sparse vectors), which replaces
  `src/aegis/infrastructure/retrieval/bm25.py` behind `LexicalIndexPort`.
- Each query makes one extra cheap call (the count) to the vector store.
- Each tenant's index reads only its own collection, so hybrid search keeps the isolation of
  [0010](0010-tenants-isolated-by-collection.md); `tests/integration/test_tenant_isolation.py`
  queries a term that exists only in the other tenant's documents.
- Turning on the reranker means building the image with `--build-arg RERANKER_MODEL=<model>`
  (about 470 MB more) so it loads offline, and setting `RETRIEVAL_MODE` to a rerank mode. It
  loads at startup, so a missing model fails the start, not a query.
- `relevance_score` is now the score of the last stage that ranked a chunk: cosine similarity,
  the fused score normalised to [0, 1], or the reranker's probability.

## Alternatives considered

- **Reranker on by default.** The best figures, but close to a second per query on CPU for two
  questions out of 38, and 470 MB in every image. Revisit with a larger evaluation set.
- **A search engine for BM25 (OpenSearch, Elasticsearch).** Another stateful service to run,
  secure and keep in sync, for corpora that fit in memory many times over.
- **Weighted score fusion instead of RRF.** Cosine similarity and BM25 live on unrelated scales,
  and BM25's grows with query length; normalising one against the other needs tuning per corpus.
  Rank fusion needs none.
- **A Spanish Snowball stemmer.** Better for Spanish, wrong for every other language the
  multilingual embedding model serves; plural stripping got most of the gain without a
  dependency.
