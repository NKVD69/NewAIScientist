"""
tests/test_v3_agents.py
Unit tests for v3.0 specific agents and Supervisor orchestration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import AnalysisAgent, ProtocolAgent, ScopingAgent, SupervisorAgent, Task, WritingAgent
from models.hypothesis import Hypothesis, ResearchGoal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_goal():
    return ResearchGoal(
        title="Test Research",
        description="Testing v3 agents",
        domain="Computer Science"
    )

def make_hypothesis():
    return Hypothesis(
        title="Test Hypothesis",
        description="Description",
        mechanism="Mechanism"
    )

# ---------------------------------------------------------------------------
# ScopingAgent
# ---------------------------------------------------------------------------

class TestScopingAgent:
    @pytest.mark.asyncio
    async def test_scoping_refinement(self):
        agent = ScopingAgent(use_local_llm=False)
        goal = make_goal()

        # Test simulated refinement
        soa = await agent.analyze_state_of_art([], goal)
        assert len(soa.known_facts) == 0
        assert "Biomedicine" not in soa.summary # Domain CS in make_goal
        assert len(soa.gaps) > 0

# ---------------------------------------------------------------------------
# ProtocolAgent
# ---------------------------------------------------------------------------

class TestProtocolAgent:
    @pytest.mark.asyncio
    async def test_protocol_generation(self):
        agent = ProtocolAgent(use_local_llm=False)
        hyp = make_hypothesis()
        goal = make_goal()

        protocol = await agent.design_experiment(hyp, goal)
        assert "Protocol" in protocol.title
        assert protocol.hypothesis_id == hyp.id

# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

class TestAnalysisAgent:
    @pytest.mark.asyncio
    async def test_results_analysis(self):
        agent = AnalysisAgent(use_local_llm=False)

        # Test exploratory analysis fallback
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        analysis = await agent.run_exploratory_analysis(df)
        assert "Summary Stats" in analysis
        assert "Missing Values" in analysis

# ---------------------------------------------------------------------------
# WritingAgent
# ---------------------------------------------------------------------------

class TestWritingAgent:
    @pytest.mark.asyncio
    async def test_manuscript_drafting(self):
        agent = WritingAgent(use_local_llm=False)
        goal = make_goal()

        section = await agent.draft_section("abstract", goal)
        assert "abstract" in section.content.lower()
        assert "requires LLM" in section.content

# ---------------------------------------------------------------------------
# LiteratureAgent
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LiteratureAgent
# ---------------------------------------------------------------------------

class TestLiteratureAgent:
    @pytest.mark.asyncio
    async def test_literature_search_fallback(self):
        from agents import LiteratureAgent
        agent = LiteratureAgent(use_local_llm=False, enable_rag=False)
        goal = make_goal()

        # Test simulated search (mocking arxiv to avoid network)
        with patch.object(agent, '_search_arxiv', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [{"title": "Cancer Study", "summary": "...", "authors": [], "url": ""}]
            papers = await agent.search_literature(goal)
            assert len(papers) > 0
            assert "Cancer" in papers[0]['title']

# ---------------------------------------------------------------------------
# ExperimentAgent
# ---------------------------------------------------------------------------

class TestExperimentAgent:
    @pytest.mark.asyncio
    async def test_experiment_safety_block(self):
        from agents import ExperimentAgent
        agent = ExperimentAgent(use_local_llm=False)
        agent.llm_client = MagicMock()

        hyp = make_hypothesis()
        goal = make_goal()

        # Dangerous code mocked as LLM output
        dangerous_code = "```python\nimport os; os.system('rm -rf /')\n```"

        with patch("agents.experiment.get_llm_completion") as mock_llm:
            mock_resp = MagicMock()
            # Set up the nested mock structure carefully
            mock_message = MagicMock()
            mock_message.content = dangerous_code
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_resp.choices = [mock_choice]
            mock_llm.return_value = mock_resp

            result = await agent.run_experiment(hyp, goal)
            assert "blocked by safety filter" in result
            assert hyp.experimental_results == result

    @pytest.mark.asyncio
    async def test_experiment_isolation(self):
        import subprocess

        from agents import ExperimentAgent
        agent = ExperimentAgent(use_local_llm=False)
        agent.llm_client = MagicMock()

        hyp = make_hypothesis()
        goal = make_goal()

        safe_code = "```python\nprint('Hello World')\n```"

        # Mock LLM and subprocess
        with patch("agents.experiment.get_llm_completion") as mock_llm, \
             patch("subprocess.run") as mock_run:

            mock_resp = MagicMock()
            mock_message = MagicMock()
            mock_message.content = safe_code
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_resp.choices = [mock_choice]
            mock_llm.return_value = mock_resp

            mock_run.return_value = MagicMock(stdout="Hello World", stderr="", returncode=0)

            await agent.run_experiment(hyp, goal)

            # Verify cwd was passed to subprocess.run and points inside the
            # OS temp tree (Windows: ...\Temp\..., Linux/macOS: /tmp/...).
            assert mock_run.called
            args, kwargs = mock_run.call_args
            assert "cwd" in kwargs
            import tempfile
            cwd_str = str(kwargs["cwd"])
            tmp_root = tempfile.gettempdir()
            assert cwd_str.startswith(tmp_root) or "temp" in cwd_str.lower() or "tmp" in cwd_str.lower(), (
                f"cwd {cwd_str!r} not under tempdir {tmp_root!r}"
            )


# ---------------------------------------------------------------------------
# SupervisorAgent (Orchestration)
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    @pytest.mark.asyncio
    async def test_task_queueing(self):
        supervisor = SupervisorAgent()
        supervisor.queue_task("Agent1", "action1", {"p1": 1}, priority=5)

        assert len(supervisor.task_queue) == 1
        task = supervisor.task_queue[0]
        assert task.priority == 5
        assert task.agent_name == "Agent1"
        assert task.action == "action1"

    @pytest.mark.asyncio
    async def test_dynamic_execution(self):
        supervisor = SupervisorAgent()

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        mock_agent.do_something = AsyncMock(return_value="done")

        supervisor.register_agent(mock_agent)
        supervisor.queue_task("MockAgent", "do_something", {"val": 123})

        await supervisor.execute_task_queue(max_iterations=1)

        mock_agent.do_something.assert_called_once_with(val=123)
        assert len(supervisor.task_history) == 1
        assert supervisor.task_history[0].result == "done"

