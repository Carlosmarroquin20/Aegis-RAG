"""
BM25 keyword index over a collection's chunks, kept in process memory.

Why in process: ChromaDB has no lexical ranking (``$contains`` filters, it does not
score), and adding a search engine to the stack for a few thousand chunks per
tenant would cost more to run than it returns. BM25 over that many chunks builds
in well under a second and answers in milliseconds. A store with native full-text
search (pgvector with ``tsvector``, OpenSearch, Qdrant sparse vectors) replaces
this module behind ``LexicalIndexPort`` once a collection outgrows it.

Freshness: the index is rebuilt from the vector store, which stays the only
source of truth. Before each search the store's chunk count is compared with the
count the index was built from, so an upload or a delete through any replica is
picked up on the next query. A change that keeps the count equal (as many chunks
deleted as added between two queries) is caught by ``max_age_seconds``.

Text normalisation is deliberately language-light, because the corpus can be in
any language the multilingual embedding model handles: Unicode folding, accents
removed, lower case, a short stop-word list for Spanish and English, and a
plural-stripping step that maps ``devoluciones``/``devolución`` and
``returns``/``return`` to the same term.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable

import structlog

from aegis.domain.models.document import Document
from aegis.domain.ports.retrieval import LexicalIndexPort
from aegis.domain.ports.vector_store import VectorStorePort

logger = structlog.get_logger(__name__)

# Robertson/Spärck Jones defaults, the same Lucene and Elasticsearch use.
K1 = 1.2
B = 0.75

_TOKEN = re.compile(r"\w+")

# Function words that carry no topic. Kept short on purpose: a long list starts
# removing words that matter in a support corpus ("sin", "no", "antes").
STOP_WORDS = frozenset(
    # Spanish (accents already removed)
    "a al algo como con cual de del donde el en es esta este esto ha hay la las le lo los "
    "me mi mis o para por que se si su sus te tu un una uno unos unas y ya yo "
    # English
    "an and are as at be by can do does for from how i if in is it its my of on or "
    "the this to was what when where which who will with you your".split()
)


def _strip_plural(token: str) -> str:
    """
    Reduces a Spanish or English plural to its singular stem, conservatively.

    ``-ces`` → ``-z`` (luces → luz); ``-ones``/``-iones`` → ``-on`` (devoluciones →
    devolucion); other ``-es`` after a consonant that Spanish pluralises that way
    (``r l n d j``) loses ``es`` (mujeres → mujer); otherwise a final ``s`` goes
    (dias → dia, returns → return). Short words and ``-ss`` are left alone, which
    keeps "mas", "les" and "class" intact.
    """
    if len(token) <= 3 or token.endswith("ss") or not token.endswith("s"):
        return token
    if token.endswith("ces") and len(token) > 4:
        return token[:-3] + "z"
    if token.endswith("ones") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and token[-3] in "rlndj":
        return token[:-2]
    return token[:-1]


def tokenize(text: str) -> list[str]:
    """Normalised index terms for ``text``, in order, stop words removed."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return [_strip_plural(token) for token in _TOKEN.findall(folded) if token not in STOP_WORDS]


class BM25Index:
    """An immutable BM25 index over a fixed set of documents. Pure CPU, no I/O."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents: list[Document] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for position, document in enumerate(documents):
            terms = Counter(tokenize(document.content))
            self._documents.append(document)
            self._lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                self._postings.setdefault(term, []).append((position, frequency))

        count = len(self._documents)
        self._average_length = (sum(self._lengths) / count) if count else 0.0
        # Lucene's form of IDF: always positive, so a term present in most chunks
        # still counts a little instead of subtracting from the score.
        self._idf = {
            term: math.log(1.0 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._documents)

    def search(self, query: str, k: int) -> list[Document]:
        """The ``k`` best-scoring documents sharing at least one term with ``query``."""
        if k < 1 or not self._documents:
            return []
        scores: dict[int, float] = {}
        # Each distinct query term counts once: repeating a word in the question
        # should not let it outweigh the others.
        for term in dict.fromkeys(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for position, frequency in postings:
                norm = K1 * (1.0 - B + B * self._lengths[position] / self._average_length)
                scores[position] = scores.get(position, 0.0) + idf * (
                    frequency * (K1 + 1.0) / (frequency + norm)
                )

        best = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        if not best:
            return []
        top = best[0][1]
        return [
            self._documents[position].model_copy(update={"relevance_score": round(score / top, 4)})
            for position, score in best
        ]


class StoreBackedLexicalIndex(LexicalIndexPort):
    """
    A ``BM25Index`` over one collection, rebuilt from its vector store when the
    store changes. One instance per collection, so a tenant's keyword search
    reads only the chunks of that tenant's collection.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        *,
        max_age_seconds: float = 300.0,
        page_size: int = 500,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self._store = vector_store
        self._max_age = max_age_seconds
        self._page_size = page_size
        self._clock = clock
        self._index: BM25Index | None = None
        self._built_from: int = -1  # the store's count when the index was built
        self._built_at: float = 0.0
        self._lock = asyncio.Lock()

    async def search(self, query: str, k: int) -> list[Document]:
        index = await self._current()
        return index.search(query, k)

    async def _current(self) -> BM25Index:
        count = await self._store.count()
        if self._fresh(count):
            assert self._index is not None
            return self._index
        async with self._lock:
            # Another query may have rebuilt it while this one waited.
            if self._fresh(count):
                assert self._index is not None
                return self._index
            started = self._clock()
            documents = await self._load(count)
            # Tokenising every chunk is CPU work; keep it off the event loop.
            self._index = await asyncio.to_thread(BM25Index, documents)
            self._built_from = count
            self._built_at = self._clock()
            logger.info(
                "lexical_index.rebuilt",
                chunks=len(self._index),
                seconds=round(self._built_at - started, 3),
            )
            return self._index

    def _fresh(self, count: int) -> bool:
        return (
            self._index is not None
            and self._built_from == count
            and self._clock() - self._built_at < self._max_age
        )

    async def _load(self, count: int) -> list[Document]:
        # Paged by the store's count rather than until a short page: a page can
        # come back short because records without text are skipped.
        documents: list[Document] = []
        for offset in range(0, count, self._page_size):
            documents.extend(await self._store.list_documents(limit=self._page_size, offset=offset))
        return documents
