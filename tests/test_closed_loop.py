"""
tests/test_closed_loop.py
Unit tests for the new closed-loop scientific discovery features:
- PreregistrationAgent
- ReplicationAgent
- ConvergenceTracker
- CoScientist integrated cycles
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import PreregistrationAgent, ReplicationAgent
from co_scientist import CoScientist
from models.hypothesis import Hypothesis, ResearchGoal, Prediction
from utils.convergence import ConvergenceTracker


def make_hypothesis(title="Test Hypothesis", testable_predictions=None) -> Hypothesis:
    h = Hypothesis(
        title=title,
        description="A test hypothesis for closed-loop validation.",
        mechanism="Test mechanism",
    )
    if testable_predictions:
        h.testable_predictions = testable_predictions
    else:
        h.testable_predictions = ["Prediction 1: target will show 20% inhibition", "Prediction 2"]
    return h


def make_goal() -> ResearchGoal:
    return ResearchGoal(
        title="Drug target validation",
        description="Validate drug targets",
        domain="Oncology",
    )


# ---------------------------------------------------------------------------
# PreregistrationAgent Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preregistration_agent_fallback():
    # Test fallback mode when LLM is unavailable
    agent = PreregistrationAgent(use_local_llm=False)
    hyp = make_hypothesis()
    goal = make_goal()

    predictions = await agent.formalize_predictions(hyp, goal)
    assert len(predictions) == 2
    assert hyp.falsifiability_score == 0.0
    assert hyp.prediction_hash != ""
    assert isinstance(predictions[0], Prediction)
    assert predictions[0].quantity == "Prediction 1: target will show 20% inhibition"


@pytest.mark.asyncio
async def test_preregistration_agent_llm():
    # Test formalizing predictions with mock LLM
    agent = PreregistrationAgent(use_local_llm=False)
    agent.llm_client = MagicMock()

    mock_llm_response = MagicMock()
    mock_llm_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""
                [
                  {
                    "quantity": "Inhibition percentage of target X",
                    "expected_value": 20.0,
                    "ci": 5.0,
                    "unit": "%",
                    "refuting_threshold": 10.0,
                    "rationale": "Prior literature shows 15-25% typical inhibition"
                  }
                ]
                """
            )
        )
    ]

    with patch("agents.preregistration.get_llm_completion", AsyncMock(return_value=mock_llm_response)):
        hyp = make_hypothesis(testable_predictions=["Pred 1"])
        goal = make_goal()
        predictions = await agent.formalize_predictions(hyp, goal)

        assert len(predictions) == 1
        assert predictions[0].quantity == "Inhibition percentage of target X"
        assert predictions[0].expected_value == 20.0
        assert predictions[0].ci == 5.0
        assert predictions[0].refuting_threshold == 10.0
        assert hyp.falsifiability_score == 1.0
        assert hyp.prediction_hash != ""


# ---------------------------------------------------------------------------
# ReplicationAgent Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replication_agent():
    # Test replication agent runs and computes reproducibility score
    agent = ReplicationAgent(use_local_llm=False)
    
    # We mock _get_experiment_code and _run_isolated to avoid full subprocess execution during tests
    agent._get_experiment_code = AsyncMock(return_value="print('p-value = 0.01; effect size = 0.5')")
    agent._run_isolated = AsyncMock(return_value="p-value = 0.01; effect size = 0.5")

    hyp = make_hypothesis()
    hyp.experimental_results = "Original run passed with p=0.01"
    goal = make_goal()

    result = await agent.replicate_experiment(hyp, goal, n_replications=2)
    assert result["reproducibility_score"] == 1.0
    assert len(result["results"]) == 2
    assert hyp.reproducibility_score == 1.0
    assert len(hyp.replication_results) == 2


# ---------------------------------------------------------------------------
# ConvergenceTracker Tests
# ---------------------------------------------------------------------------

def test_convergence_tracker_stable_elo():
    tracker = ConvergenceTracker(elo_threshold=10.0, patience=1)
    
    h1 = make_hypothesis()
    h1.elo_rating = 1200.0
    h1.novelty_level = "medium"

    hypotheses = {h1.id: h1}

    # Iteration 1: Initial check
    report1 = tracker.update(hypotheses, [], iteration=1)
    assert not report1.converged

    # Iteration 2: Elo doesn't change, novelty plateaued
    report2 = tracker.update(hypotheses, [], iteration=2)
    assert report2.converged
    assert tracker.should_stop(2)


# ---------------------------------------------------------------------------
# CoScientist Integrated Cycles Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coscientist_integrated_cycles():
    # Test CoScientist integrated workflow methods
    co = CoScientist(use_local_llm=False, enable_rag=False)
    
    # Pre-populate hypotheses
    h = make_hypothesis()
    co.context_memory.hypotheses[h.id] = h
    co.context_memory.research_goal = make_goal()

    # Verify preregistration cycle
    await co.run_preregistration_cycle()
    assert len(h.falsifiable_predictions) > 0
    assert h.prediction_hash != ""

    # Mock experiment and replication execution to prevent actual process execution
    co.replication_agent.replicate_experiment = AsyncMock(return_value={
        "reproducibility_score": 1.0,
        "results": ["p-value = 0.01"]
    })

    # Verify replication cycle runs on top hypotheses
    h.elo_rating = 1500.0  # Make it the top hypothesis
    h.experimental_results = "Experiment passed"
    await co.run_replication_cycle()
    co.replication_agent.replicate_experiment.assert_called_once()

    # Verify revision cycle triggers when prediction is refuted
    h.experimental_results = "Experiment failed and prediction refuted"
    co.evolution_agent.evolve_hypothesis = AsyncMock(return_value=make_hypothesis("Revised Hypothesis"))
    
    revised = await co.run_revision_cycle()
    assert len(revised) == 1
    assert revised[0].title == "Revised Hypothesis"
