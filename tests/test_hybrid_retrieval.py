"""
tests/test_hybrid_retrieval.py
Offline tests for hybrid retrieval primitives.
The cross-encoder is exercised in fallback mode (no model load),
so these tests run without sentence-transformers installed.
"""

from __future__ import annotations

import pytest

from utils.hybrid_retrieval import (
    BM25Index,
    CrossEncoderReranker,
    hybrid_search,
    reciprocal_rank_fusion,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []

    def test_none_safe(self):
        assert tokenize(None) == []  # type: ignore[arg-type]

    def test_unicode_word_chars(self):
        # \w matches Unicode word chars under default re.UNICODE
        assert tokenize("alpha β-catenin") == ["alpha", "β", "catenin"]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

class TestRRF:
    def test_single_ranking(self):
        out = reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [doc for doc, _ in out]
        assert ids == ["a", "b", "c"]

    def test_two_rankings_top_overlap_wins(self):
        out = reciprocal_rank_fusion([["a", "b", "c"], ["a", "x", "y"]])
        ids = [doc for doc, _ in out]
        # 'a' appears at rank 0 in both lists ⇒ highest fused score
        assert ids[0] == "a"

    def test_disjoint_rankings(self):
        out = reciprocal_rank_fusion([["a"], ["b"]])
        ids = [doc for doc, _ in out]
        # Both at rank 0 → equal score → both present
        assert sorted(ids) == ["a", "b"]

    def test_k_parameter_dampens_lower_ranks(self):
        out_high_k = reciprocal_rank_fusion([["a", "b", "c"]], k=1000)
        out_low_k = reciprocal_rank_fusion([["a", "b", "c"]], k=1)
        scores_high = dict(out_high_k)
        scores_low = dict(out_low_k)
        # Lower k → larger gap between rank 0 and rank 2
        gap_high = scores_high["a"] - scores_high["c"]
        gap_low = scores_low["a"] - scores_low["c"]
        assert gap_low > gap_high

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_empty_index_returns_no_results(self):
        idx = BM25Index()
        idx.build()
        assert idx.search("anything") == []
        assert not idx.is_ready or len(idx) == 0

    def test_basic_search_when_lib_available(self):
        pytest.importorskip("rank_bm25")
        idx = BM25Index()
        idx.add("doc1", "machine learning models for protein folding")
        idx.add("doc2", "cooking recipes for pasta and tomato sauce")
        idx.add("doc3", "deep learning for protein structure prediction")
        idx.build()
        results = idx.search("protein folding learning", top_k=3)
        ids = [r[0] for r in results]
        assert ids[0] in {"doc1", "doc3"}
        assert "doc2" not in ids[:1]

    def test_unbuilt_index_returns_empty(self):
        idx = BM25Index()
        idx.add("doc1", "some text")
        # No build() called
        assert idx.search("text") == []

    def test_add_invalidates_built_index(self):
        pytest.importorskip("rank_bm25")
        idx = BM25Index()
        idx.add("doc1", "first")
        idx.build()
        assert idx.is_ready
        idx.add("doc2", "second")
        # After add, the cached BM25 instance must be invalidated.
        assert not idx.is_ready


# ---------------------------------------------------------------------------
# Cross-encoder reranker — fallback path (no model)
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerFallback:
    def test_no_candidates(self):
        rr = CrossEncoderReranker()
        assert rr.rerank("q", [], top_k=5) == []

    def test_fallback_preserves_input_order(self):
        # Force the failed flag so we go through the no-model branch
        rr = CrossEncoderReranker()
        rr._failed = True
        candidates = [("a", "txt a"), ("b", "txt b"), ("c", "txt c")]
        out = rr.rerank("q", candidates, top_k=2)
        assert [doc_id for doc_id, _ in out] == ["a", "b"]
        # Scores monotonic-decreasing
        assert out[0][1] > out[1][1]


# ---------------------------------------------------------------------------
# hybrid_search orchestrator
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def _dense(self):
        return [
            ("c1", "machine learning models"),
            ("c2", "deep neural networks"),
            ("c3", "irrelevant pasta cooking"),
        ]

    def test_dense_only_when_no_bm25(self):
        out = hybrid_search(
            "machine learning",
            dense_results=self._dense(),
            bm25=None,
            top_k=2,
        )
        ids = [doc_id for doc_id, _, _ in out]
        assert ids == ["c1", "c2"]

    def test_top_k_respected(self):
        out = hybrid_search(
            "machine learning",
            dense_results=self._dense(),
            bm25=None,
            top_k=1,
        )
        assert len(out) == 1

    def test_text_passthrough(self):
        out = hybrid_search(
            "ml",
            dense_results=self._dense(),
            bm25=None,
            top_k=3,
        )
        texts = {doc_id: text for doc_id, _, text in out}
        assert texts["c1"] == "machine learning models"

    def test_with_bm25_promotes_keyword_match(self):
        pytest.importorskip("rank_bm25")
        # Dense ranking puts the cooking doc last.
        # BM25 with the same query also disfavours it, so c1/c2 stay on top.
        idx = BM25Index()
        idx.add("c1", "machine learning models")
        idx.add("c2", "deep neural networks")
        idx.add("c3", "pasta cooking recipes")
        idx.build()
        out = hybrid_search(
            "machine learning",
            dense_results=self._dense(),
            bm25=idx,
            top_k=2,
        )
        ids = [doc_id for doc_id, _, _ in out]
        assert "c3" not in ids

    def test_rerank_called_when_reranker_provided(self):
        # Use a fallback reranker (failed) — must still produce a stable order.
        rr = CrossEncoderReranker()
        rr._failed = True
        out = hybrid_search(
            "ml",
            dense_results=self._dense(),
            bm25=None,
            top_k=2,
            reranker=rr,
        )
        assert len(out) == 2
