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
from models.hypothesis import Hypothesis, Prediction, ResearchGoal
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
async def test_replication_agent_runs_multiverse():
    """Replication now varies analytic specifications, not random seeds.

    Seed variation measured nothing: on real data the analysis is
    deterministic (variance zero by construction), and on synthetic data it
    measured the variance of the LLM's own RNG.
    """
    from utils import sandbox_runner

    agent = ReplicationAgent(use_local_llm=False, max_specifications=8)
    agent._get_experiment_code = AsyncMock(return_value="multiverse_analyse(a, b)")

    call_count = {"n": 0}

    def fake_container(runtime, policy, host_dir):
        call_count["n"] += 1
        effect = 0.5 if call_count["n"] % 4 else -0.2   # 1 in 4 flips sign
        return sandbox_runner.SandboxResult(
            stdout='SPEC_RESULT:{"effect": %s, "p_value": 0.01, "n": 60}' % effect,
            exit_code=0, backend=runtime,
        )

    with patch.object(sandbox_runner, "detect_runtime", lambda *a, **k: "docker"), \
         patch.object(sandbox_runner, "_run_container", fake_container):
        hyp, goal = make_hypothesis(), make_goal()
        result = await agent.replicate_experiment(hyp, goal)

    assert call_count["n"] == 8, "every specification must be executed"
    assert 0.0 < result["reproducibility_score"] < 1.0
    assert hyp.multiverse_fragility > 0.0
    assert len(hyp.replication_results) == 8


@pytest.mark.asyncio
async def test_fragility_feeds_the_ranking_prior():
    """A fragile finding must be penalised in the tournament prior."""
    from agents.ranking import RankingAgent

    agent = RankingAgent(use_local_llm=False, verify_citations=False)
    robust = make_hypothesis("robust"); robust.multiverse_fragility = 0.05
    fragile = make_hypothesis("fragile"); fragile.multiverse_fragility = 0.95

    assert agent.get_rating(robust).mu > agent.get_rating(fragile).mu


@pytest.mark.asyncio
async def test_replication_without_code_is_honest_about_it():
    agent = ReplicationAgent(use_local_llm=False)
    agent._get_experiment_code = AsyncMock(return_value="")
    result = await agent.replicate_experiment(make_hypothesis(), make_goal())
    assert result["reproducibility_score"] == 0.0
    assert "not performed" in result["consistency"]


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

    # Revision is now driven by adjudicated verdicts, not by grepping stdout.
    # The old version of this test set
    #     h.experimental_results = "Experiment failed and prediction refuted"
    # and expected a revision -- i.e. it asserted the substring-matching bug.
    _attach_run(h, refuted=[("IC50", 2.0, 47.3, "uM")])
    co.evolution_agent.evolve_hypothesis = AsyncMock(return_value=make_hypothesis("Revised Hypothesis"))

    revised = await co.run_revision_cycle()
    assert len(revised) == 1
    assert revised[0].title == "Revised Hypothesis"

    # The refuted quantity must reach the evolution agent, not an opaque blob.
    _, kwargs = co.evolution_agent.evolve_hypothesis.call_args
    assert kwargs["strategy"] == "experimental_revision"
    assert kwargs["refutations"][0]["quantity"] == "IC50"


# ---------------------------------------------------------------------------
# Regression tests: the grep-based refutation detector
# ---------------------------------------------------------------------------

def _attach_run(
    hyp: Hypothesis,
    refuted: list[tuple] | None = None,
    corroborated: list[tuple] | None = None,
    stdout: str = "",
):
    """Attach a structurally-adjudicated ExperimentRun to a hypothesis.

    Each tuple is ``(quantity, expected, observed, unit)``. Registers the
    matching predictions, seals them with a valid hash so the anti-HARKing
    gate passes, and runs the real adjudicator.
    """
    from models.experiment import ExperimentKind, ExperimentRun, Measurement
    from utils.adjudication import adjudicate

    refuted = refuted or []
    corroborated = corroborated or []

    hyp.falsifiable_predictions = [
        Prediction(quantity=q, expected_value=exp, ci=0.1, unit=unit,
                   refuting_threshold=1.0)
        for q, exp, _obs, unit in (refuted + corroborated)
    ]
    hyp.prediction_hash = PreregistrationAgent._compute_prediction_hash(
        hyp.falsifiable_predictions
    )

    measurements = [
        Measurement(quantity=q, observed=obs, unit=unit)
        for q, _exp, obs, unit in (refuted + corroborated)
    ]

    run = ExperimentRun(
        hypothesis_id=hyp.id,
        kind=ExperimentKind.REAL_DATA_ANALYSIS,
        measurements=measurements,
        stdout=stdout,
    )
    run.verdicts = adjudicate(hyp, measurements, kind=ExperimentKind.REAL_DATA_ANALYSIS)

    hyp.experiment_runs.append(run.to_dict())
    hyp.verdicts = [v.to_dict() for v in run.verdicts]
    hyp.experimental_results = stdout
    return run


async def _revision_cycle_with(hyp: Hypothesis) -> list:
    co = CoScientist(use_local_llm=False, enable_rag=False)
    co.context_memory.hypotheses[hyp.id] = hyp
    co.context_memory.research_goal = make_goal()
    co.evolution_agent.evolve_hypothesis = AsyncMock(
        return_value=make_hypothesis("Revised")
    )
    return await co.run_revision_cycle()


@pytest.mark.asyncio
async def test_failed_to_reject_does_not_trigger_revision():
    """'failed to reject the null' contains both 'fail' and 'reject'.

    The old detector fired on it twice over, treating the standard phrase for
    *absence of evidence against* as a refutation. The measurement here agrees
    with the registered prediction, so no revision must occur.
    """
    h = make_hypothesis()
    _attach_run(
        h,
        corroborated=[("IC50", 2.0, 2.1, "uM")],
        stdout="Welch t-test: p=0.68. We failed to reject the null hypothesis.",
    )
    assert "fail" in h.experimental_results.lower()      # old detector would fire
    assert "reject" in h.experimental_results.lower()

    revised = await _revision_cycle_with(h)
    assert revised == []


@pytest.mark.asyncio
async def test_quantitative_refutation_without_trigger_words_is_caught():
    """A refutation the old substring detector would have missed entirely."""
    h = make_hypothesis()
    _attach_run(
        h,
        refuted=[("IC50", 2.0, 47.3, "uM")],
        stdout="Measured half-maximal inhibitory concentration: 47.3 uM.",
    )
    low = h.experimental_results.lower()
    assert not any(t in low for t in ("fail", "not support", "refute", "reject"))

    revised = await _revision_cycle_with(h)
    assert len(revised) == 1


@pytest.mark.asyncio
async def test_untested_predictions_do_not_trigger_revision_nor_count_as_support():
    """Silence is not evidence, in either direction."""
    from models.experiment import ExperimentKind, ExperimentRun, VerdictStatus
    from utils.adjudication import adjudicate

    h = make_hypothesis()
    h.falsifiable_predictions = [
        Prediction(quantity="IC50", expected_value=2.0, ci=0.1, unit="uM",
                   refuting_threshold=1.0)
    ]
    h.prediction_hash = PreregistrationAgent._compute_prediction_hash(
        h.falsifiable_predictions
    )
    run = ExperimentRun(hypothesis_id=h.id, kind=ExperimentKind.REAL_DATA_ANALYSIS)
    run.verdicts = adjudicate(h, [], kind=ExperimentKind.REAL_DATA_ANALYSIS)
    h.experiment_runs.append(run.to_dict())
    h.verdicts = [v.to_dict() for v in run.verdicts]

    assert h.verdicts[0]["status"] == VerdictStatus.UNTESTED.value
    assert run.evidential_weight == 0.0

    revised = await _revision_cycle_with(h)
    assert revised == []


@pytest.mark.asyncio
async def test_harked_predictions_are_rejected_before_adjudication():
    """Predictions moved after registration invalidate their own verdicts."""
    h = make_hypothesis()
    _attach_run(h, refuted=[("IC50", 2.0, 47.3, "uM")])

    # Move the goalposts so the "refutation" now looks like a success.
    h.falsifiable_predictions[0].expected_value = 47.0

    revised = await _revision_cycle_with(h)
    assert revised == []
    assert any("integrity failure" in lim for lim in h.limitations)


@pytest.mark.asyncio
async def test_simulation_agreement_does_not_become_support():
    """A simulation agreeing with itself must not raise empirical support."""
    from models.experiment import ExperimentKind, ExperimentRun, Measurement, VerdictStatus
    from utils.adjudication import adjudicate

    h = make_hypothesis()
    h.falsifiable_predictions = [
        Prediction(quantity="effect", expected_value=0.5, ci=0.05, unit="",
                   refuting_threshold=0.2)
    ]
    measurements = [Measurement(quantity="effect", observed=0.51)]
    run = ExperimentRun(hypothesis_id=h.id, kind=ExperimentKind.DRY_RUN_SIMULATION,
                        measurements=measurements)
    run.verdicts = adjudicate(h, measurements, kind=ExperimentKind.DRY_RUN_SIMULATION)

    assert run.verdicts[0].status is VerdictStatus.CONSISTENT_UNSCORED
    assert run.evidential_weight == 0.0
    assert run.n_corroborated == 0
