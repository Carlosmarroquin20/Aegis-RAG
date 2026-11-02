"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from aegis.domain.models.document import Document
from aegis.domain.services.rank_fusion import reciprocal_rank_fusion


def _doc(doc_id: str, score: float = 0.5) -> Document:
    return Document(id=doc_id, content=f"content of {doc_id}", relevance_score=score)


def _ids(documents: list[Document]) -> list[str]:
    return [d.id for d in documents]


class TestReciprocalRankFusion:
    def test_a_document_high_in_both_rankings_wins(self) -> None:
        dense = [_doc("a"), _doc("b"), _doc("c")]
        lexical = [_doc("c"), _doc("b"), _doc("d")]

        fused = reciprocal_rank_fusion([dense, lexical])

        # b is second in both; a and c are first in one and absent or third in the other.
        assert _ids(fused)[0] in {"b", "c"}
        assert set(_ids(fused)) == {"a", "b", "c", "d"}

    def test_documents_are_identified_by_id_and_appear_once(self) -> None:
        fused = reciprocal_rank_fusion([[_doc("a"), _doc("b")], [_doc("a")]])

        assert _ids(fused) == ["a", "b"]

    def test_first_everywhere_scores_one(self) -> None:
        fused = reciprocal_rank_fusion([[_doc("a"), _doc("b")], [_doc("a")]])

        assert fused[0].relevance_score == 1.0
        assert 0.0 < fused[1].relevance_score < 1.0

    def test_scores_are_the_fused_ones_not_the_inputs(self) -> None:
        fused = reciprocal_rank_fusion([[_doc("a", score=0.99)], [_doc("b", score=0.01)]])

        # Both are first in one ranking out of two: equal fused scores.
        assert fused[0].relevance_score == fused[1].relevance_score == 0.5

    def test_ties_keep_first_seen_order(self) -> None:
        fused = reciprocal_rank_fusion([[_doc("a")], [_doc("b")]])

        assert _ids(fused) == ["a", "b"]

    def test_a_smaller_k_rewards_the_top_more(self) -> None:
        """With k=1 a single first place beats two middling places; with k=60 it does not."""
        # k=1:  solo 1/2 = 0.50   both 1/5 + 1/4 = 0.45
        # k=60: solo 1/61 ≈ 0.016 both 1/64 + 1/63 ≈ 0.031
        rankings = [
            [_doc("solo"), _doc("x1"), _doc("x2"), _doc("both")],
            [_doc("x3"), _doc("x4"), _doc("both")],
        ]

        assert _ids(reciprocal_rank_fusion(rankings, k=1))[0] == "solo"
        assert _ids(reciprocal_rank_fusion(rankings, k=60))[0] == "both"

    def test_empty_rankings_fuse_to_nothing(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_one_empty_ranking_does_not_lower_the_scale(self) -> None:
        """An empty lexical list leaves the dense order intact and scored out of one ranking."""
        fused = reciprocal_rank_fusion([[_doc("a"), _doc("b")], []])

        assert _ids(fused) == ["a", "b"]
        assert fused[0].relevance_score == 1.0

    def test_k_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[_doc("a")]], k=0)
