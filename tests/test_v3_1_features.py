"""
tests/test_v3_1_features.py — Unit tests for v3.1 functional enhancements.
"""

import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import config
from utils.llm import get_llm_completion, get_llm_usage_stats
from agents.literature import LiteratureAgent
from agents.writing import WritingAgent
from utils.notebook_exporter import generate_reproducible_notebook
from models.hypothesis import Hypothesis, ResearchGoal


def test_config_role_based_llm():
    os.environ["MODEL_RAG"] = "fast-rag-model"
    assert config.get_llm_model_name_for_role("rag") == "fast-rag-model"
    assert config.get_llm_model_name_for_role("reasoning") == config.get_llm_model_name()
    del os.environ["MODEL_RAG"]


@pytest.mark.asyncio
async def test_llm_role_completion():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "ok"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.total_tokens = 150
    mock_client.chat.completions.create.return_value = mock_response

    res = await get_llm_completion(mock_client, [{"role": "user", "content": "hi"}], agent_role="reasoning")
    assert res is not None
    stats = get_llm_usage_stats()
    assert stats["total_tokens"] >= 150
    assert "reasoning" in stats["role_tokens"]


@pytest.mark.asyncio
async def test_literature_openalex_search():
    agent = LiteratureAgent(use_local_llm=False, enable_rag=False)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "results": [
                {
                    "display_name": "Test Paper",
                    "publication_date": "2026-01-01",
                    "authorships": [{"author": {"display_name": "Dr. Smith"}}],
                    "doi": "https://doi.org/10.1234/test",
                    "cited_by_count": 42
                }
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        results = await agent._search_openalex("leukemia", max_results=2)
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"
        assert results[0]["source"] == "OpenAlex"


def test_notebook_exporter(tmp_path):
    mock_cs = MagicMock()
    mock_cs.context_memory.research_goal = ResearchGoal(
        title="AML Repurposing", description="Test description", domain="Oncology"
    )
    mock_hyp = Hypothesis(
        id="h1",
        title="Drug X for AML",
        description="Mechanism explanation",
        mechanism="Target pathway Y",
        elo_rating=1350.0,
        novelty_level="high",
        experimental_results="print('Test code output')"
    )
    mock_cs.context_memory.hypotheses = {"h1": mock_hyp}

    out_file = str(tmp_path / "test_notebook.ipynb")
    nb_path = generate_reproducible_notebook(mock_cs, out_file)
    assert os.path.exists(nb_path)

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 4
