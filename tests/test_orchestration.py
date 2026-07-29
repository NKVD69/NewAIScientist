"""
tests/test_orchestration.py
Integration-style tests for the CoScientist orchestrator.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from co_scientist import CoScientist
from utils.pipeline import FailurePolicy, PipelineReport
from models import Hypothesis, ResearchGoal


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
    async def test_run_full_cycle_executes_declared_dags(self, co_scientist):
        """run_full_cycle now runs validated task DAGs, not a priority heap.

        Asserting on the DAG rather than on queue_task calls is a stronger
        test: it checks the dependency structure, which is what actually
        prevents downstream tasks running on a corrupted state.
        """
        co_scientist.supervisor.run_pipeline = AsyncMock(
            return_value=PipelineReport()
        )

        await co_scientist.run_full_cycle(num_iterations=1)

        executed = [
            spec.action
            for call in co_scientist.supervisor.run_pipeline.call_args_list
            for spec in call.args[0]
        ]
        for action in (
            "run_literature_search", "run_scoping_cycle",
            "run_hypothesis_generation_cycle", "run_tournament_cycle",
            "run_writing_cycle",
        ):
            assert action in executed, f"{action} was never scheduled"

    def test_pipelines_declare_a_valid_acyclic_graph(self, co_scientist):
        from utils.pipeline import validate

        for specs in (
            co_scientist._initial_pipeline(),
            co_scientist._iteration_pipeline(1),
            co_scientist._validation_pipeline(),
            co_scientist._output_pipeline(),
        ):
            validate(specs)   # raises PipelineError on cycles/unknown deps

    def test_generation_depends_on_literature(self, co_scientist):
        """The dependency whose absence produced ungrounded hypotheses."""
        specs = {s.name: s for s in co_scientist._initial_pipeline()}
        assert "literature" in specs["generation"].depends_on
        assert specs["generation"].on_failure is FailurePolicy.ABORT

    def test_independent_tasks_share_a_wave(self, co_scientist):
        from utils.pipeline import topological_waves

        waves = topological_waves(co_scientist._iteration_pipeline(1))
        first = set(waves[0])
        assert {"review", "proximity"} <= first, "independent tasks must parallelise"
