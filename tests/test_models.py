"""
tests/test_models.py
Unit tests for data model classes (Hypothesis, ResearchGoal, ContextMemory, TournamentMatch).
These tests are fully offline — no LLM or network required.
"""

from dataclasses import asdict

import pytest

from models.hypothesis import Hypothesis, HypothesisStatus, ResearchGoal, ReviewCritique
from models.memory import ContextMemory, TournamentMatch

# ---------------------------------------------------------------------------
# HypothesisStatus
# ---------------------------------------------------------------------------

class TestHypothesisStatus:
    def test_all_status_values(self):
        expected = {"generated", "under_review", "reviewed", "in_tournament", "ranked", "evolved", "completed"}
        actual = {s.value for s in HypothesisStatus}
        assert actual == expected

    def test_enum_comparison(self):
        assert HypothesisStatus.GENERATED != HypothesisStatus.REVIEWED


# ---------------------------------------------------------------------------
# ReviewCritique
# ---------------------------------------------------------------------------

class TestReviewCritique:
    def test_default_timestamp(self):
        r = ReviewCritique(
            review_type="llm-full",
            correctness_score=0.8,
            novelty_score=0.7,
            testability_score=0.9,
            quality_score=0.8,
            feedback="OK",
        )
        assert r.timestamp  # Non-empty ISO timestamp

    def test_scores_are_floats(self):
        r = ReviewCritique("test", 1, 0, 0.5, 0.75, "feedback")
        assert isinstance(r.correctness_score, int | float)
        assert isinstance(r.novelty_score, int | float)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    def test_default_elo(self):
        h = Hypothesis()
        assert h.elo_rating == 1200.0

    def test_unique_ids(self):
        ids = {Hypothesis().id for _ in range(50)}
        assert len(ids) == 50, "All IDs should be unique"

    def test_default_status(self):
        h = Hypothesis()
        assert h.status == HypothesisStatus.GENERATED

    def test_asdict_serializable(self):
        h = Hypothesis(title="test", description="desc")
        d = asdict(h)
        assert d["title"] == "test"
        assert d["elo_rating"] == 1200.0

    def test_reviews_list(self):
        h = Hypothesis()
        r = ReviewCritique("test", 0.5, 0.5, 0.5, 0.5, "ok")
        h.reviews.append(r)
        assert len(h.reviews) == 1

    def test_reviews_are_not_shared(self):
        """Each Hypothesis instance must have its own review list (no shared default)."""
        h1 = Hypothesis()
        h2 = Hypothesis()
        h1.reviews.append(ReviewCritique("t", 0.5, 0.5, 0.5, 0.5, "x"))
        assert len(h2.reviews) == 0

    def test_testable_predictions_not_shared(self):
        h1 = Hypothesis()
        h2 = Hypothesis()
        h1.testable_predictions.append("pred")
        assert len(h2.testable_predictions) == 0


# ---------------------------------------------------------------------------
# ResearchGoal
# ---------------------------------------------------------------------------

class TestResearchGoal:
    def test_default_constraints_empty(self):
        g = ResearchGoal(title="Test", description="desc", domain="Physics")
        assert g.constraints == []

    def test_preferences_not_shared(self):
        g1 = ResearchGoal()
        g2 = ResearchGoal()
        g1.preferences["key"] = "val"
        assert "key" not in g2.preferences


# ---------------------------------------------------------------------------
# TournamentMatch
# ---------------------------------------------------------------------------

class TestTournamentMatch:
    def test_match_ids_unique(self):
        matches = [TournamentMatch() for _ in range(20)]
        ids = {m.match_id for m in matches}
        assert len(ids) == 20

    def test_field_defaults(self):
        m = TournamentMatch(hypothesis_a_id="a", hypothesis_b_id="b", winner_id="a")
        assert m.winner_id == "a"
        assert m.debate_summary == ""


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------

class TestContextMemory:
    def test_empty_on_init(self):
        mem = ContextMemory()
        assert mem.hypotheses == {}
        assert mem.tournament_history == []
        assert mem.iteration_count == 0

    def test_add_hypothesis(self):
        mem = ContextMemory()
        h = Hypothesis(title="New hypothesis")
        mem.hypotheses[h.id] = h
        assert h.id in mem.hypotheses
        assert mem.hypotheses[h.id].title == "New hypothesis"

    def test_literature_context_isolated(self):
        mem1 = ContextMemory()
        mem2 = ContextMemory()
        mem1.literature_context.append({"title": "Paper"})
        assert mem2.literature_context == []
