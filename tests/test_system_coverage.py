"""
tests/test_system_coverage.py
Additional tests to improve coverage for utils/llm.py and rag_system.py.
"""

import asyncio
import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from utils.llm import get_llm_completion, parse_json_response, ensure_str, get_llm_usage_stats
from rag_system import RAGEngine

# ---------------------------------------------------------------------------
# utils/llm.py Tests
# ---------------------------------------------------------------------------

class TestLLMUtils:
    @pytest.mark.asyncio
    async def test_get_llm_completion_retry_success(self):
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.usage.total_tokens = 10
        # Fail once, then succeed
        client.chat.completions.create.side_effect = [
            Exception("API Error"),
            mock_resp
        ]
        
        with patch("utils.llm.cfg.get_llm_model_name", return_value="gpt-4"):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                resp = await get_llm_completion(client, [{"role": "user", "content": "hi"}], max_retries=2)
                assert resp == mock_resp
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_get_llm_completion_json_fallback(self):
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.usage.total_tokens = 5
        
        # Fail with JSON mode error, then succeed without it
        client.chat.completions.create.side_effect = [
            Exception("400: response_format of type 'json_object' is not supported"),
            mock_resp
        ]
        
        with patch("utils.llm.cfg.get_llm_model_name", return_value="legacy-model"):
            resp = await get_llm_completion(client, [{"role": "user", "content": "hi"}], json_mode=True)
            assert resp == mock_resp
            # Verify the second call didn't have response_format
            last_args = client.chat.completions.create.call_args.kwargs
            assert "response_format" not in last_args

    def test_parse_json_response_variants(self):
        # 1. Simple JSON
        assert parse_json_response('{"a": 1}') == {"a": 1}
        # 2. Markdown fence
        assert parse_json_response('```json\n{"b": 2}\n```') == {"b": 2}
        # 3. Simple fence
        assert parse_json_response('```\n{"c": 3}\n```') == {"c": 3}
        # 4. Text around JSON
        assert parse_json_response('Here is the data: {"d": 4} end.') == {"d": 4}
        # 5. Invalid JSON
        with pytest.raises(json.JSONDecodeError):
            parse_json_response('no json here')

    def test_ensure_str(self):
        assert ensure_str(None) == ""
        assert ensure_str("hi") == "hi"
        assert ensure_str(["a", "b"]) == "a b"
        assert ensure_str(123) == "123"

# ---------------------------------------------------------------------------
# rag_system.py Tests
# ---------------------------------------------------------------------------

class TestRAGEngine:
    @pytest.fixture
    def mock_rag(self):
        with patch("rag_system.SentenceTransformer"), \
             patch("rag_system.chromadb.PersistentClient"):
            engine = RAGEngine()
            engine.collection = MagicMock()
            # Mock get() to return empty results so it doesn't skip "already indexed" papers
            engine.collection.get.return_value = {"ids": []}
            return engine

    @pytest.mark.asyncio
    async def test_process_papers(self, mock_rag):
        papers = [{"title": "P1", "summary": "Summary of P1", "url": "http://p1"}]
        
        # Mock dependencies to avoid network/file IO
        mock_rag.downloader.download_paper = AsyncMock(return_value="/tmp/p1.txt")
        mock_rag.processor.extract_text = AsyncMock(return_value="Extracted text content from paper.")
        mock_rag.embedding_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]*384])
        
        count = await mock_rag.process_papers(papers)
        assert count > 0
        assert mock_rag.collection.add.called

    @pytest.mark.asyncio
    async def test_query(self, mock_rag):
        mock_rag.collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"paper_title": "P1", "paper_id": "ID1"}, {"paper_title": "P2", "paper_id": "ID2"}]],
            "distances": [[0.1, 0.2]]
        }
        mock_rag.embedding_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]*384])
        
        results = await mock_rag.query("test query", top_k=2)
        assert len(results) == 2
        assert results[0]["paper_title"] == "P1"
        assert results[1]["text"] == "Doc 2"
