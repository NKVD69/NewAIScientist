"""
tests/test_interactive_feedback.py
Tests for UserFeedback model, the CLI feedback driver, and
EvolutionAgent.evolve_with_feedback.
"""

from __future__ import annotations

import asyncio

import pytest

from agents.evolution import EvolutionAgent
from models.hypothesis import Hypothesis, UserFeedback
from utils.interactive_feedback import collect_feedback_cli

# ---------------------------------------------------------------------------
# UserFeedback model
# ---------------------------------------------------------------------------

class TestUserFeedbackModel:
    def test_default_verdict_is_refine(self):
        fb = UserFeedback(hypothesis_id="abc")
        assert fb.verdict == "refine"

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValueError):
            UserFeedback(hypothesis_id="abc", verdict="maybe")

    def test_valid_verdicts(self):
        for v in ("agree", "disagree", "refine"):
            fb = UserFeedback(hypothesis_id="x", verdict=v)
            assert fb.verdict == v


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _make_hyps(n: int = 3):
    return [
        Hypothesis(id=f"h{i}", title=f"Hypothesis {i}", elo_rating=1500.0 - i * 10)
        for i in range(n)
    ]


def _scripted_prompt(answers):
    """Return a callable that yields successive scripted answers."""
    queue = list(answers)

    def _prompt(_msg):
        return queue.pop(0)
    return _prompt


def _silent(_msg):
    return None


class TestCollectFeedbackCLI:
    def test_skip_all_returns_empty(self):
        prompt = _scripted_prompt(["s", "s", "s"])
        out = collect_feedback_cli(_make_hyps(3), prompt=prompt, output=_silent)
        assert out == []

    def test_agree_disagree_refine_mix(self):
        prompt = _scripted_prompt([
            "a",                     # h0: agree
            "d",                     # h1: disagree
            "r", "tighten the scope",  # h2: refine + comment
        ])
        out = collect_feedback_cli(_make_hyps(3), prompt=prompt, output=_silent)
        assert [fb.verdict for fb in out] == ["agree", "disagree", "refine"]
        assert out[2].comment == "tighten the scope"
        assert out[2].hypothesis_id == "h2"

    def test_invalid_then_valid(self):
        prompt = _scripted_prompt([
            "lol", "wat", "agree",   # h0: 2 invalid then agree
            "s",                     # h1: skip
            "s",                     # h2: skip
        ])
        out = collect_feedback_cli(_make_hyps(3), prompt=prompt, output=_silent)
        assert len(out) == 1
        assert out[0].verdict == "agree"

    def test_refine_with_empty_comment_is_skipped(self):
        prompt = _scripted_prompt([
            "r", "",   # h0: refine with empty comment ⇒ treated as skip
            "s",       # h1
            "s",       # h2
        ])
        out = collect_feedback_cli(_make_hyps(3), prompt=prompt, output=_silent)
        assert out == []

    def test_aliases(self):
        prompt = _scripted_prompt(["yes", "no", "edit", "shorter"])
        out = collect_feedback_cli(_make_hyps(3), prompt=prompt, output=_silent)
        assert [fb.verdict for fb in out] == ["agree", "disagree", "refine"]


# ---------------------------------------------------------------------------
# EvolutionAgent.evolve_with_feedback (LLM disabled — fallback path)
# ---------------------------------------------------------------------------

class TestEvolveWithFeedback:
    def _agent(self):
        # use_local_llm=False ⇒ no llm_client is configured (covered by BaseAgent)
        agent = EvolutionAgent(use_local_llm=False)
        agent.llm_client = None
        return agent

    def test_agree_returns_none(self):
        agent = self._agent()
        hyp = Hypothesis(title="H", mechanism="m")
        fb = UserFeedback(hypothesis_id=hyp.id, verdict="agree")
        result = asyncio.run(agent.evolve_with_feedback(hyp, fb))
        assert result is None

    def test_disagree_returns_none(self):
        agent = self._agent()
        hyp = Hypothesis(title="H")
        fb = UserFeedback(hypothesis_id=hyp.id, verdict="disagree", comment="bad framing")
        result = asyncio.run(agent.evolve_with_feedback(hyp, fb))
        assert result is None

    def test_refine_creates_child_with_constraint(self):
        agent = self._agent()
        hyp = Hypothesis(
            id="orig", title="Drug X inhibits Y", mechanism="binds active site",
        )
        fb = UserFeedback(
            hypothesis_id=hyp.id,
            verdict="refine",
            comment="must specify the dose-response curve",
        )
        result = asyncio.run(agent.evolve_with_feedback(hyp, fb))

        assert result is not None
        assert result.id != hyp.id
        assert "orig" in result.parent_ids
        assert result.generation_method == "evolved-feedback"
        # The constraint must be preserved in the limitations.
        assert any("dose-response curve" in lim for lim in result.limitations)
        # Counter increments
        assert agent.evolved_hypotheses == 1

    def test_refine_increments_counter_only_on_actual_evolution(self):
        agent = self._agent()
        hyp = Hypothesis(id="orig", title="H")
        # Three feedbacks; only one produces a child.
        asyncio.run(agent.evolve_with_feedback(
            hyp, UserFeedback(hypothesis_id="orig", verdict="agree"),
        ))
        asyncio.run(agent.evolve_with_feedback(
            hyp, UserFeedback(hypothesis_id="orig", verdict="disagree"),
        ))
        asyncio.run(agent.evolve_with_feedback(
            hyp, UserFeedback(hypothesis_id="orig", verdict="refine", comment="x"),
        ))
        assert agent.evolved_hypotheses == 1
