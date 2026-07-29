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
            # The AST check is now a *quality* filter, not a security boundary
            # (see utils/safety.py and utils/sandbox_runner.py). The wording
            # changed to stop callers treating a pass as authorisation to run.
            assert "quality filter" in result
            assert "os" in result
            assert hyp.experimental_results == result

    @pytest.mark.asyncio
    async def test_execution_is_refused_without_isolation(self, monkeypatch):
        """Execution must fail closed, not fall back to the user's privileges.

        Replaces the old test, which asserted that ``subprocess.run`` was
        called with ``cwd`` pointing into the temp tree. That property was
        never isolation: cwd changes the working directory, it does not
        confine the filesystem, and the script still ran with the full user
        UID and unrestricted network. Execution now goes through
        ``utils.sandbox_runner``, which refuses to run at all when no
        container runtime is available.
        """
        from agents import ExperimentAgent
        from utils import sandbox_runner

        monkeypatch.setattr(sandbox_runner, "detect_runtime", lambda *a, **k: "none")

        agent = ExperimentAgent(use_local_llm=False)
        agent.llm_client = MagicMock()
        hyp, goal = make_hypothesis(), make_goal()

        with patch("agents.experiment.get_llm_completion") as mock_llm:
            mock_resp = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "```python\nprint('Hello World')\n```"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_resp.choices = [mock_choice]
            mock_llm.return_value = mock_resp

            result = await agent.run_experiment(hyp, goal)

        assert "execution refused" in result.lower()
        assert hyp.experiment_runs, "the refused run must still be recorded"
        assert hyp.experiment_runs[-1]["exit_code"] is None

    @pytest.mark.asyncio
    async def test_sandboxed_run_is_recorded_structurally(self, monkeypatch):
        """A successful run yields an ExperimentRun, not just a text blob."""
        from agents import ExperimentAgent
        from utils import sandbox_runner
        from utils.adjudication import RESULTS_MARKER

        stdout = (
            "Analysis complete.\n"
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "IC50", '
            '"observed": 2.1, "unit": "uM"}]}'
        )
        monkeypatch.setattr(sandbox_runner, "detect_runtime", lambda *a, **k: "docker")
        monkeypatch.setattr(
            sandbox_runner, "_run_container",
            lambda runtime, policy, host_dir: sandbox_runner.SandboxResult(
                stdout=stdout, exit_code=0, backend=runtime, duration_s=0.1,
            ),
        )

        agent = ExperimentAgent(use_local_llm=False)
        agent.llm_client = MagicMock()
        hyp, goal = make_hypothesis(), make_goal()

        with patch("agents.experiment.get_llm_completion") as mock_llm:
            mock_resp = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "```python\nprint('ok')\n```"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_resp.choices = [mock_choice]
            mock_llm.return_value = mock_resp

            await agent.run_experiment(hyp, goal)

        assert len(hyp.experiment_runs) == 1
        run = hyp.experiment_runs[0]
        assert run["sandbox_backend"] == "docker"
        assert run["measurements"][0]["quantity"] == "IC50"
        assert run["code_sha256"]


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

