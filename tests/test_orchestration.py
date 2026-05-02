"""
tests/test_orchestration.py
Integration-style tests for the CoScientist orchestrator.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from co_scientist import CoScientist
from models import ResearchGoal, Hypothesis

class TestCoScientistOrchestration:
    @pytest.fixture
    def co_scientist(self):
        with patch("co_scientist.GenerationAgent"), \
             patch("co_scientist.LiteratureAgent"), \
             patch("co_scientist.ScopingAgent"), \
             patch("co_scientist.ProtocolAgent"), \
             patch("co_scientist.AnalysisAgent"), \
             patch("co_scientist.WritingAgent"), \
             patch("co_scientist.SupervisorAgent"):
            return CoScientist(use_local_llm=False)

    @pytest.mark.asyncio
    async def test_initialize_goal(self, co_scientist):
        goal = await co_scientist.initialize_research_goal(
            title="Test Goal",
            description="Test Description",
            domain="Test Domain"
        )
        assert goal.title == "Test Goal"
        assert co_scientist.context_memory.research_goal == goal

    @pytest.mark.asyncio
    async def test_run_scoping_cycle(self, co_scientist):
        from models.hypothesis import StateOfArt
        # Use real dataclass instead of MagicMock for asdict compatibility
        soa_mock = StateOfArt(
            known_facts=["fact"],
            gaps=["gap"],
            contradictions=[],
            summary="summary"
        )
        
        co_scientist.scoping_agent.analyze_state_of_art = AsyncMock(return_value=soa_mock)
        co_scientist.scoping_agent.generate_research_questions = AsyncMock(return_value=[])
        co_scientist.scoping_agent.build_conceptual_framework = AsyncMock(return_value={})
        
        co_scientist.context_memory.literature_context = []
        co_scientist.context_memory.research_goal = ResearchGoal(title="T", description="D", domain="D")
        
        result = await co_scientist.run_scoping_cycle()
        assert "soa" in result
        assert co_scientist.scoping_agent.analyze_state_of_art.called

    @pytest.mark.asyncio
    async def test_run_full_workflow_mocked(self, co_scientist):
        """Test that run_full_cycle queues the right tasks and executes them."""
        # Mock supervisor to avoid full execution but check queueing
        co_scientist.supervisor.queue_task = MagicMock()
        co_scientist.supervisor.execute_task_queue = AsyncMock()
        
        await co_scientist.run_full_cycle(num_iterations=1)
        
        # Verify initial tasks queued
        calls = [call.args[1] for call in co_scientist.supervisor.queue_task.call_args_list]
        assert "run_literature_search" in calls
        assert "run_scoping_cycle" in calls
        assert "run_hypothesis_generation_cycle" in calls
        assert "run_writing_cycle" in calls
        
        assert co_scientist.supervisor.execute_task_queue.called
