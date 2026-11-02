"""
Reciprocal rank fusion (Cormack, Clarke and Büttcher, SIGIR 2009).

Merges rankings produced by different retrievers into one. Each document scores
the sum of ``1 / (k + rank)`` over the rankings it appears in, so a chunk that is
near the top of both the dense and the lexical list beats one that tops only one
of them.

It uses ranks, not scores, on purpose: cosine similarities and BM25 scores live on
unrelated scales (BM25 is unbounded and grows with query length), and normalising
them against each other is fragile in a way that ranks are not. ``k`` damps the
weight of the very first positions; 60 is the value from the paper and works well
without tuning.
"""

from __future__ import annotations

from collections.abc import Sequence

from aegis.domain.models.document import Document

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]], *, k: int = DEFAULT_RRF_K
) -> list[Document]:
    """
    Fuses ``rankings`` (each best first) into one list, best first.

    Documents are identified by id. The returned copies carry the fused score,
    divided by the best score possible (first in every ranking), so it falls in
    [0, 1] like every other ``relevance_score``: 1.0 means first everywhere.
    Ties keep the order in which documents were first seen, so the result is
    deterministic.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            first_seen.setdefault(document.id, document)
            scores[document.id] = scores.get(document.id, 0.0) + 1.0 / (k + rank)

    if not scores:
        return []

    best_possible = sum(1.0 / (k + 1) for ranking in rankings if ranking)
    order = sorted(first_seen, key=lambda doc_id: -scores[doc_id])
    return [
        first_seen[doc_id].model_copy(
            update={"relevance_score": round(min(1.0, scores[doc_id] / best_possible), 4)}
        )
        for doc_id in order
    ]
