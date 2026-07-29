"""
tests/test_agents.py
Unit tests for agent classes with mocked LLM clients.
Fully offline — LLM is stubbed out.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from co_scientist import (
    EvolutionAgent,
    MetaReviewAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
)
from models.hypothesis import Hypothesis, HypothesisStatus, ResearchGoal, ReviewCritique
from models.memory import ContextMemory, TournamentMatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hypothesis(title="Test Hyp", mechanism="inhibits KRAS pathway via FTI") -> Hypothesis:
    h = Hypothesis(title=title, mechanism=mechanism)
    h.testable_predictions = ["Pred 1", "Pred 2"]
    h.grounding_evidence = ["Evidence A"]
    return h


def make_goal() -> ResearchGoal:
    return ResearchGoal(
        title="Drug repurposing for AML",
        description="Find drugs for AML",
        domain="Biomedicine",
    )


# ---------------------------------------------------------------------------
# ProximityAgent — vector similarity
# ---------------------------------------------------------------------------

class TestProximityAgent:
    """ProximityAgent should compute similarity between hypotheses."""

    def test_same_hypothesis_max_similarity(self):
        agent = ProximityAgent()
        # Disable embedding model to test Jaccard fallback
        agent._embedding_model = None
        h = make_hypothesis()

        sim = asyncio.run(agent._compute_similarity(h, h))
        # Self-similarity should be 1.0 (identical mechanism words)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_different_hypotheses_lower_similarity(self):
        agent = ProximityAgent()
        agent._embedding_model = None

        h1 = make_hypothesis("H1", "KRAS inhibition reduces tumor growth significantly")
        h2 = make_hypothesis("H2", "immune checkpoint blockade activates T-cell response")

        sim_self = asyncio.run(agent._compute_similarity(h1, h1))
        sim_cross = asyncio.run(agent._compute_similarity(h1, h2))
        assert sim_self > sim_cross

    def test_compute_proximity_returns_map(self):
        agent = ProximityAgent()
        agent._embedding_model = None
        hypotheses = [make_hypothesis(f"H{i}") for i in range(3)]

        prox = asyncio.run(agent.compute_proximity(hypotheses))
        assert len(prox) == 3
        for _hyp_id, neighbors in prox.items():
            # Should have n-1 neighbors
            assert len(neighbors) == 2
            # Neighbors sorted by similarity (descending)
            if len(neighbors) >= 2:
                assert neighbors[0][1] >= neighbors[1][1]


# ---------------------------------------------------------------------------
# RankingAgent — Elo update
# ---------------------------------------------------------------------------

class TestRankingAgent:
    """RankingAgent should update Elo ratings correctly."""

    def test_elo_updates_on_win(self):
        agent = RankingAgent(use_local_llm=False)
        h_a = make_hypothesis("Strong")
        h_b = make_hypothesis("Weak")
        h_a.elo_rating = 1200.0
        h_b.elo_rating = 1200.0

        # Manually update Elo with h_a winning
        agent._update_elo_ratings(h_a, h_b, h_a.id)

        assert h_a.elo_rating > 1200.0
        assert h_b.elo_rating < 1200.0

    def test_rating_update_is_not_zero_sum_by_design(self):
        """Bradley-Terry deliberately breaks Elo's sum conservation.

        Under Elo both competitors move by the same amount. Under BT the
        better-observed competitor (smaller sigma) moves less, because we
        already had good evidence about it. Sum conservation is therefore
        not an invariant here -- asserting it would forbid the very property
        that makes uncertainty load-bearing.
        """
        agent = RankingAgent(use_local_llm=False)
        h_a = make_hypothesis("well-observed")
        h_b = make_hypothesis("newcomer")
        h_a.rating_mu, h_a.rating_sigma, h_a.rating_matches = 1200.0, 60.0, 20
        h_b.rating_mu, h_b.rating_sigma, h_b.rating_matches = 1200.0, 200.0, 0

        agent._update_elo_ratings(h_a, h_b, h_a.id)

        moved_a = abs(h_a.rating_mu - 1200.0)
        moved_b = abs(h_b.rating_mu - 1200.0)
        assert moved_b > moved_a, "the less certain belief must move more"

    def test_conduct_match_with_fallback(self):
        """Without LLM, tournament should use heuristic fallback."""
        agent = RankingAgent(use_local_llm=False)
        h_a = make_hypothesis("A")
        h_b = make_hypothesis("B")

        winner_id, match = asyncio.run(agent.conduct_tournament_match(h_a, h_b))
        assert winner_id in {h_a.id, h_b.id}
        assert match.winner_id == winner_id

    def test_debate_score_increases_with_predictions(self):
        agent = RankingAgent(use_local_llm=False)
        h_few = make_hypothesis("Few")
        h_few.testable_predictions = ["P1"]
        h_many = make_hypothesis("Many")
        h_many.testable_predictions = ["P1", "P2", "P3", "P4"]

        assert agent._compute_debate_score(h_many) > agent._compute_debate_score(h_few)


# ---------------------------------------------------------------------------
# ReflectionAgent — simulated review
# ---------------------------------------------------------------------------

class TestReflectionAgent:
    def test_simulated_review_returns_critique(self):
        agent = ReflectionAgent(use_local_llm=False)
        h = make_hypothesis()
        goal = make_goal()

        review = asyncio.run(agent._review_simulated(h, goal))
        assert isinstance(review, ReviewCritique)
        assert 0.0 <= review.correctness_score <= 1.0
        assert 0.0 <= review.novelty_score <= 1.0
        assert 0.0 <= review.quality_score <= 1.0

    def test_novelty_level_updated(self):
        agent = ReflectionAgent(use_local_llm=False)
        h = make_hypothesis()
        goal = make_goal()

        asyncio.run(agent._review_simulated(h, goal))
        assert h.novelty_level in {"low", "medium", "high", "very_high"}


# ---------------------------------------------------------------------------
# MetaReviewAgent
# ---------------------------------------------------------------------------

class TestMetaReviewAgent:
    def test_generate_meta_review_structure(self):
        agent = MetaReviewAgent()
        hypotheses = [make_hypothesis(f"H{i}") for i in range(5)]
        goal = make_goal()
        history = []

        mr = asyncio.run(agent.generate_meta_review(hypotheses, history, goal))
        assert "total_hypotheses" in mr
        assert mr["total_hypotheses"] == 5
        assert "top_hypotheses" in mr
        assert "suggested_improvements" in mr
        assert "research_overview" in mr

    def test_top_hypotheses_sorted_by_elo(self):
        agent = MetaReviewAgent()
        goal = make_goal()
        hypotheses = [make_hypothesis(f"H{i}") for i in range(5)]
        for i, h in enumerate(hypotheses):
            h.elo_rating = 1200.0 + i * 50  # H4 has highest Elo

        mr = asyncio.run(agent.generate_meta_review(hypotheses, [], goal))
        top = mr["top_hypotheses"]
        assert top[0]["elo_rating"] >= top[-1]["elo_rating"]


# ---------------------------------------------------------------------------
# EvolutionAgent
# ---------------------------------------------------------------------------

class TestEvolutionAgent:
    def test_evolve_creates_child(self):
        agent = EvolutionAgent(use_local_llm=False)
        original = make_hypothesis("Original")

        child = asyncio.run(agent.evolve_hypothesis(original, strategy="enhancement"))
        assert child.id != original.id
        assert original.id in child.parent_ids

    def test_evolve_increments_counter(self):
        agent = EvolutionAgent(use_local_llm=False)
        original = make_hypothesis()
        initial = agent.evolved_hypotheses

        asyncio.run(agent.evolve_hypothesis(original))
        assert agent.evolved_hypotheses == initial + 1
