# 0023 — Multilingual E5 embeddings, with their query and passage markers

**Status:** Accepted. Supersedes the model choice of
[0015](0015-multilingual-embeddings-recorded-per-collection.md); the per-collection model check
stands.

## Context

On the 141 labelled questions ([0021](0021-larger-evaluation-sets.md)), paraphrases were where
retrieval failed. With `paraphrase-multilingual-MiniLM-L12-v2`, hybrid retrieval found the right
document in the top 5 for 90.3% of them, with MRR 0.747. Direct questions reached 98.7% and 0.912.
Two things decide how well a reworded question matches its document: the embedding model and the
size of the chunks it embeds. Both were measured in CI on the same questions, through the
application's own ingestion and retrieval code.

Several current multilingual retrieval models are asymmetric. The E5 family is trained with
`query: ` before a question and `passage: ` before a stored text, and loses accuracy without them.
ChromaDB 1.x embeds query texts through an embedding function's `embed_query`, so the two sides
can be marked differently.

## Measurements

Embedding models at the current chunk size (512 characters), on a CI runner's CPU:

| Model | Mode | hit@1 | hit@5 | MRR | Paraphrases hit@5 / MRR | per query | Weights |
|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 *(before)* | hybrid | 75.9% | 96.5% | 0.850 | 90.3% / 0.747 | 20 ms | ~470 MB |
| paraphrase-multilingual-mpnet-base-v2 | hybrid | 76.6% | 97.9% | 0.861 | 93.5% / 0.833 | 53 ms | ~1.1 GB |
| **multilingual-e5-small** | **hybrid** | **82.3%** | **97.2%** | **0.882** | 90.3% / 0.811 | **19 ms** | **~470 MB** |
| multilingual-e5-small | dense | 81.6% | 97.2% | 0.877 | 90.3% / 0.757 | 19 ms | ~470 MB |
| multilingual-e5-base | dense | 85.1% | 98.6% | 0.903 | 96.8% / 0.907 | 56 ms | ~1.1 GB |
| bge-m3 | hybrid | 89.4% | 99.3% | 0.933 | 96.8% / 0.891 | 188 ms | ~2.3 GB |
| **bge-m3** | **dense** | **92.2%** | **100%** | **0.957** | **100% / 0.910** | 180 ms | ~2.3 GB |

Chunk sizes with the previous model, hybrid:

| Chunk size | Chunks | hit@1 | hit@5 | MRR | Paraphrases hit@5 |
|---|---|---|---|---|---|
| 256 | 51 | 78.0% | 95.0% | 0.860 | 83.9% |
| 384 | 36 | 77.3% | 94.3% | 0.858 | 80.6% |
| **512** *(kept)* | 26 | 75.9% | 96.5% | 0.850 | 90.3% |
| 768 | 18 | 70.9% | 97.2% | 0.823 | 90.3% |
| 1024 | 14 | 69.5% | 97.9% | 0.814 | 90.3% |

## Decision

- **The default embedding model becomes `intfloat/multilingual-e5-small`.** It is the same size
  and speed as the model it replaces, and it raises MRR from 0.850 to 0.882 and hit@1 from 75.9% to
  82.3%. The gain comes from Spanish questions (MRR 0.870 to 0.921), scenarios (0.797 to 0.888)
  and paraphrases (0.747 to 0.811). Direct questions hold: MRR 0.912 to 0.907, with all of them now
  in the top 5.
- **Chunks and searches carry the model's markers.**
  `PromptedSentenceTransformerEmbeddingFunction` subclasses ChromaDB's SentenceTransformers
  function. It embeds documents with `passage: ` and queries, through `embed_query`, with
  `query: `. It keeps the parent's `name()` and configuration, so a collection records the same
  function type. The markers come from the model name (`embedding_prompts.py`), and a symmetric
  model gets none. The evaluation embeds through the same table. An integration test checks, in
  an in-process ChromaDB, that stored documents and query texts each get their own marker.
- **`RETRIEVAL_MODE` stays `hybrid`.** With e5-small, hybrid beats dense slightly (0.882 against
  0.877).
- **`BAAI/bge-m3` with `RETRIEVAL_MODE=dense` is the documented option for the best retrieval.**
  It found the right document in the top 5 for every question, including every paraphrase and
  every English one. It also beats the cross-encoder reranker (MRR 0.957 against 0.910) at a fifth
  of its latency.
- **Chunks stay at 512 characters.** No size improved everything. Smaller chunks put more
  documents first but lose them from the top 5, especially paraphrases (83.9% at 256). Larger
  chunks do the opposite. The top 5 is what the model reads, so the current size stays.

## Consequences

- **Upgrading needs a decision from the operator.** A collection built with the previous model was
  embedded in an incompatible space. The check from 0015 makes the API refuse to start on it, with
  a message that names both models. Either set
  `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to keep the
  existing collections, or re-ingest the documents into a new collection. The CHANGELOG says so.
- **Paraphrases improve in rank, not in reach.** With e5-small, more paraphrases are answered from
  the first chunk (hit@1 61.3% to 74.2%), but 3 of 31 still fall outside the top 5, as before
  (a different three).
  e5-base (96.8%) or bge-m3 (100%) close that gap, at roughly three and nine times the latency on
  CPU. That cost is the reason they are options rather than the default.
- **English is where the small model gains least.** English MRR moves from 0.819 to 0.822,
  against 0.921 for Spanish. With every model, hybrid helps Spanish questions and costs English
  ones a little, because BM25 cannot match English words in a Spanish corpus. With the strongest
  models dense beats hybrid outright: bge-m3 scores 0.957 dense against 0.933 hybrid. That is why
  the bge-m3 option is dense.
- **The reranker matters less.** On e5-small it adds 0.028 MRR (0.882 to 0.910) and 2 questions in
  the top 5, at about a second per query. bge-m3 alone does better.
- **The retrieval gate rises** to hit@5 0.96 and MRR 0.86, just under the e5-small figures (hit@5
  0.972 and MRR 0.882), on 0012's principle that one more miss passes and two fail.
- **The image does not grow.** e5-small replaces a model of the same size, and the Dockerfile still
  bakes whichever `EMBEDDING_MODEL` the build names.

## Alternatives considered

- **bge-m3 as the default.** It is the best measured by a wide margin. But it adds about 1.8 GB to
  the image and about 2 GB of memory, and takes ~180 ms per query on CPU against 19 ms. A
  deployment that wants it changes two settings and rebuilds. A default that triples the image does
  not suit someone trying the stack on a laptop.
- **e5-base.** It is between the two on every axis. Paraphrases reach 96.8% at ~56 ms per query.
  It is a reasonable middle setting, but nothing justified making it the default over the
  measured gains of e5-small at no cost.
- **Models that need `trust_remote_code`** (gte-multilingual, jina-embeddings-v3). Loading
  executable code with the weights is ruled out, as for every other model in the service.
- **Query expansion or HyDE with the LLM.** Rewriting the question before retrieval helps
  paraphrases, but it adds an LLM call per query and makes retrieval non-deterministic, which a CI
  gate cannot measure reliably.
