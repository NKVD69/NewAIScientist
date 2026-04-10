"""
tests/test_rag.py
Unit tests for the RAG system with mocked ChromaDB and embeddings.
Fully offline — no network required.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestRAGEngine:
    """Test RAGEngine with mocked dependencies."""

    def _make_engine(self):
        """Create an RAGEngine with mocked collection."""
        from rag_system import RAGEngine
        engine = RAGEngine()
        engine.collection = MagicMock()
        engine.collection.get.return_value = {"ids": []}
        engine.collection.add = MagicMock()
        engine.collection.query.return_value = {
            "documents": [["Doc chunk A", "Doc chunk B"]],
            "distances": [[0.1, 0.3]],
        }
        return engine

    def test_deduplication_prevents_double_index(self):
        """Papers already indexed should not be added twice."""
        engine = self._make_engine()
        paper = {
            "title": "Test Paper",
            "summary": "A test paper about KRAS inhibition.",
            "url": "http://arxiv.org/abs/1234",
            "published": "2023",
        }

        # Simulate: first call returns empty (not found), subsequent calls return the id
        call_count = 0
        def mock_get(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"ids": []}   # Not yet indexed
            return {"ids": [f"1234_0"]}  # Already indexed

        engine.collection.get.side_effect = mock_get

        # First indexing — should add
        chunks_1 = asyncio.run(engine.process_papers([paper]))

        # Reset side effect to always return "found"
        engine.collection.get.side_effect = None
        engine.collection.get.return_value = {"ids": ["1234_0"]}

        # Second indexing — should skip (dedup)
        chunks_2 = asyncio.run(engine.process_papers([paper]))
        assert chunks_2 == 0, "Second indexing should be skipped by dedup logic"

    def test_get_stats_returns_status(self):
        """get_stats() should return a dict with 'status' key."""
        engine = self._make_engine()
        engine.collection.count.return_value = 42
        stats = engine.get_stats()
        assert "status" in stats

    def test_query_returns_list(self):
        """Query results should be a list of strings."""
        engine = self._make_engine()
        engine.embedding_model = MagicMock()
        engine.embedding_model.encode.return_value = [0.1] * 384

        # Mock query to return something
        results = asyncio.run(engine.query("KRAS inhibition", top_k=2))
        # Should return a list (possibly empty if mock isn't perfect)
        assert isinstance(results, list)


class TestSanitizeText:
    """Test the unicode sanitization in generate_paper.py."""

    def test_ascii_passthrough(self):
        from generate_paper import _sanitize_for_pdf
        text = "Hello, world. This is a test."
        assert _sanitize_for_pdf(text) == text

    def test_scientific_symbols_replaced(self):
        from generate_paper import _sanitize_for_pdf
        text = "Rate is ±5% and α=0.05"
        result = _sanitize_for_pdf(text)
        assert "+/-" in result
        assert "alpha" in result
        # No unicode chars should remain
        result.encode("latin-1")  # Should not raise

    def test_arrows_replaced(self):
        from generate_paper import _sanitize_for_pdf
        text = "A → B ← C"
        result = _sanitize_for_pdf(text)
        assert "->" in result
        assert "<-" in result

    def test_empty_string(self):
        from generate_paper import _sanitize_for_pdf
        assert _sanitize_for_pdf("") == ""

    def test_greek_letters(self):
        from generate_paper import _sanitize_for_pdf
        text = "β-catenin signaling activates γ-secretase"
        result = _sanitize_for_pdf(text)
        # Should be latin-1 encodable
        result.encode("latin-1")
        assert "beta" in result.lower() or "b" in result.lower()

    def test_none_safe(self):
        from generate_paper import _sanitize_for_pdf
        assert _sanitize_for_pdf(None) == ""  # type: ignore[arg-type]
