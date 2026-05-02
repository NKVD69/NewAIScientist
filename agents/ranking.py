"""
agents/ranking.py — RankingAgent for tournament-based hypothesis ranking.

Responsible for:
- Pairwise hypothesis comparison via LLM-as-judge
- Elo rating system updates
- Debate summary generation
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from models.hypothesis import Hypothesis
from models.memory import TournamentMatch
from utils.citation_verifier import (
    verify_hypothesis,
    verification_score,
)
from utils.llm import ensure_str, get_llm_completion, parse_json_response
from .base import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """Tournament-based hypothesis ranking using Elo system with LLM-as-judge."""

    name = "Ranking"

    # Default weights for the multi-criterion judge.
    DEFAULT_CRITERIA_WEIGHTS = {
        "novelty": 0.25,
        "plausibility": 0.30,
        "testability": 0.25,
        "impact": 0.20,
    }

    def __init__(
        self,
        use_local_llm: bool = True,
        verify_citations: bool = True,
        criteria_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(use_local_llm=use_local_llm)
        self.k_factor = 32  # Elo K-factor
        self.matches_completed = 0
        # When True, citation verification adjusts the per-match Elo update.
        self.verify_citations = verify_citations
        # Cached score per hypothesis ID to avoid re-hitting the network.
        self._citation_cache: Dict[str, float] = {}
        # Multi-criterion judge configuration. Weights re-normalise to sum=1.
        self.criteria_weights = self._normalise_weights(
            criteria_weights or self.DEFAULT_CRITERIA_WEIGHTS,
        )
        # Multi-dimensional Elo per hypothesis: per-criterion ratings.
        # populated lazily by _multi_judge.
        self.multi_elo: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, v) for v in weights.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in weights.items()}

    def per_criterion_rating(self, hyp_id: str, criterion: str) -> float:
        """Return the per-criterion Elo (default 1200) for a hypothesis."""
        return self.multi_elo.get(hyp_id, {}).get(criterion, 1200.0)

    async def _citation_score(self, hyp: Hypothesis) -> float:
        """Return the cached fraction of resolved citations for *hyp* (1.0 if disabled)."""
        if not self.verify_citations:
            return 1.0
        if hyp.id in self._citation_cache:
            return self._citation_cache[hyp.id]
        try:
            results = await verify_hypothesis(hyp)
            score = verification_score(results)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Citation verification failed for %s: %s", hyp.id, exc)
            score = 1.0
        self._citation_cache[hyp.id] = score
        return score
    
    async def conduct_tournament_match(self,
                                      hyp_a: Hypothesis,
                                      hyp_b: Hypothesis) -> Tuple[str, TournamentMatch]:
        """
        Conduct pairwise hypothesis comparison through simulated scientific debate.
        Returns winner ID and match record.
        """
        winner_id = await self._simulate_debate(hyp_a, hyp_b)
        debate_summary = self._generate_debate_summary(hyp_a, hyp_b, winner_id)

        # Citation verification — fold into the Elo update
        cit_a = await self._citation_score(hyp_a)
        cit_b = await self._citation_score(hyp_b)

        # Update Elo ratings (modulated by citation trust)
        self._update_elo_ratings(hyp_a, hyp_b, winner_id, cit_a, cit_b)
        
        match = TournamentMatch(
            hypothesis_a_id=hyp_a.id,
            hypothesis_b_id=hyp_b.id,
            winner_id=winner_id,
            debate_summary=debate_summary
        )
        
        self.matches_completed += 1
        return winner_id, match
    
    async def _simulate_debate(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> str:
        """
        Determine the winning hypothesis via multi-criterion LLM-as-judge
        (primary) or score-based heuristic with randomness (fallback).
        """
        if self.llm_client:
            try:
                verdicts = await self._multi_judge(hyp_a, hyp_b)
                if verdicts:
                    self._update_multi_elo(hyp_a, hyp_b, verdicts)
                    return self._aggregate_verdicts(hyp_a.id, hyp_b.id, verdicts)
            except Exception as e:
                logger.warning("LLM multi-judge failed, falling back: %s", e)

            # Legacy single-criterion path as a secondary fallback
            try:
                winner_id = await self._llm_debate(hyp_a, hyp_b)
                if winner_id:
                    return winner_id
            except Exception as e:
                logger.warning("LLM debate failed, falling back to heuristic: %s", e)

        # Heuristic fallback: score + noise
        score_a = self._compute_debate_score(hyp_a)
        score_b = self._compute_debate_score(hyp_b)
        debate_factor_a = random.uniform(0.85, 1.15)
        debate_factor_b = random.uniform(0.85, 1.15)
        return hyp_a.id if score_a * debate_factor_a > score_b * debate_factor_b else hyp_b.id

    async def _multi_judge(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
    ) -> Dict[str, str]:
        """Single LLM call returning per-criterion winners.

        Returns a dict ``{criterion: winning_id}`` covering at least one of
        the four criteria. Empty dict if the call/parse fails.
        """
        criteria = list(self.criteria_weights.keys())
        prompt = (
            "You are a senior scientific reviewer comparing two research hypotheses.\n"
            f"Score each on these four criteria: {', '.join(criteria)}.\n\n"
            f"Hypothesis A (ID: {hyp_a.id}):\n"
            f"- Title: {hyp_a.title}\n"
            f"- Mechanism: {ensure_str(hyp_a.mechanism)[:300]}\n"
            f"- Predictions: {', '.join(hyp_a.testable_predictions[:3])}\n\n"
            f"Hypothesis B (ID: {hyp_b.id}):\n"
            f"- Title: {hyp_b.title}\n"
            f"- Mechanism: {ensure_str(hyp_b.mechanism)[:300]}\n"
            f"- Predictions: {', '.join(hyp_b.testable_predictions[:3])}\n\n"
            "Definitions:\n"
            "- novelty: how much new ground does the hypothesis break?\n"
            "- plausibility: how consistent is it with established science?\n"
            "- testability: how concretely can it be falsified by experiment?\n"
            "- impact: how meaningful would the outcome be if true?\n\n"
            "Return ONLY raw JSON of the form:\n"
            '{"verdicts": {"novelty": "<A or B>", "plausibility": "<A or B>", '
            '"testability": "<A or B>", "impact": "<A or B>"}, '
            '"reasoning": "<one short sentence>"}'
        )
        response = await get_llm_completion(
            self.llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            json_mode=True,
        )
        data = parse_json_response(response.choices[0].message.content)
        raw = data.get("verdicts") or {}
        out: Dict[str, str] = {}
        for crit in criteria:
            v = str(raw.get(crit, "")).strip().upper()
            if v == "A" or v == hyp_a.id:
                out[crit] = hyp_a.id
            elif v == "B" or v == hyp_b.id:
                out[crit] = hyp_b.id
        return out

    def _aggregate_verdicts(
        self,
        a_id: str,
        b_id: str,
        verdicts: Dict[str, str],
    ) -> str:
        """Reduce per-criterion verdicts to a single winner via weighted vote."""
        score_a = sum(
            self.criteria_weights.get(c, 0.0) for c, w in verdicts.items() if w == a_id
        )
        score_b = sum(
            self.criteria_weights.get(c, 0.0) for c, w in verdicts.items() if w == b_id
        )
        # Tie ⇒ pick A deterministically (first listed).
        return a_id if score_a >= score_b else b_id

    def _update_multi_elo(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        verdicts: Dict[str, str],
    ) -> None:
        """Update per-criterion Elo ratings stored in self.multi_elo."""
        for criterion, winner in verdicts.items():
            ra = self.per_criterion_rating(hyp_a.id, criterion)
            rb = self.per_criterion_rating(hyp_b.id, criterion)
            ea = 1.0 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 - ea
            sa = 1.0 if winner == hyp_a.id else 0.0
            sb = 1.0 - sa
            self.multi_elo.setdefault(hyp_a.id, {})[criterion] = ra + self.k_factor * (sa - ea)
            self.multi_elo.setdefault(hyp_b.id, {})[criterion] = rb + self.k_factor * (sb - eb)

    async def _llm_debate(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> Optional[str]:
        """Use the LLM as scientific judge to compare two hypotheses."""
        prompt = f"""
        You are a senior scientific reviewer adjudicating a research hypothesis competition.
        Compare the following two hypotheses and decide which is more scientifically promising.

        Hypothesis A (ID: {hyp_a.id}):
        - Title: {hyp_a.title}
        - Mechanism: {ensure_str(hyp_a.mechanism)[:300]}
        - Predictions: {', '.join(hyp_a.testable_predictions[:3])}
        - Novelty: {hyp_a.novelty_level}

        Hypothesis B (ID: {hyp_b.id}):
        - Title: {hyp_b.title}
        - Mechanism: {ensure_str(hyp_b.mechanism)[:300]}
        - Predictions: {', '.join(hyp_b.testable_predictions[:3])}
        - Novelty: {hyp_b.novelty_level}

        Evaluate on: scientific rigor, novelty, experimental feasibility, and mechanistic specificity.
        Return JSON: {{"winner_id": "<id of A or B>", "reasoning": "brief 1-sentence justification"}}
        """
        response = await get_llm_completion(
            self.llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            json_mode=True
        )
        data = parse_json_response(response.choices[0].message.content)
        raw_id = data.get("winner_id", "")
        if raw_id == hyp_a.id or raw_id.upper() == "A":
            logger.debug("LLM judge chose A (%s): %s", hyp_a.id, data.get('reasoning', ''))
            return hyp_a.id
        elif raw_id == hyp_b.id or raw_id.upper() == "B":
            logger.debug("LLM judge chose B (%s): %s", hyp_b.id, data.get('reasoning', ''))
            return hyp_b.id
        return None
    
    def _compute_debate_score(self, hypothesis: Hypothesis) -> float:
        score = 0.5
        if hypothesis.reviews:
            avg_review_score = sum(
                r.novelty_score * 0.3 + r.testability_score * 0.3 + r.correctness_score * 0.2 + r.quality_score * 0.2
                for r in hypothesis.reviews
            ) / len(hypothesis.reviews)
            score = avg_review_score
        score += len(hypothesis.testable_predictions) * 0.05
        novelty_bonus = {
            "very_high": 0.15, "high": 0.10, "medium": 0.05, "low": 0.00, "unknown": 0.02
        }
        score += novelty_bonus.get(hypothesis.novelty_level, 0.02)
        return min(1.0, max(0.0, score))
    
    def _update_elo_ratings(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        winner_id: str,
        cit_a: float = 1.0,
        cit_b: float = 1.0,
    ):
        """Standard Elo update, with the winner's gain damped by its citation score.

        ``cit_a`` / ``cit_b`` are in [0, 1]. A winner whose citations are all
        hallucinated still gets points (we don't want zero signal) but at half
        the rate, while the loser's drop is amplified accordingly. The reverse
        applies when the loser is the one with shaky citations.
        """
        expected_a = 1 / (1 + 10 ** ((hyp_b.elo_rating - hyp_a.elo_rating) / 400))
        expected_b = 1 - expected_a

        # Trust factor: 0.5 + 0.5 * citation_score, so worst case = half points.
        trust_a = 0.5 + 0.5 * max(0.0, min(1.0, cit_a))
        trust_b = 0.5 + 0.5 * max(0.0, min(1.0, cit_b))

        if winner_id == hyp_a.id:
            hyp_a.elo_rating += self.k_factor * trust_a * (1 - expected_a)
            hyp_b.elo_rating += self.k_factor * trust_b * (0 - expected_b)
        else:
            hyp_a.elo_rating += self.k_factor * trust_a * (0 - expected_a)
            hyp_b.elo_rating += self.k_factor * trust_b * (1 - expected_b)
    
    def _generate_debate_summary(self, hyp_a: Hypothesis, hyp_b: Hypothesis, winner_id: str) -> str:
        winner = hyp_a if winner_id == hyp_a.id else hyp_b
        loser = hyp_b if winner_id == hyp_a.id else hyp_a
        summary = f"Debate winner: {winner.title[:50]}... "
        summary += f"(Elo: {winner.elo_rating:.0f}) defeated "
        summary += f"{loser.title[:50]}... (Elo: {loser.elo_rating:.0f}). "
        if winner.novelty_level == "very_high":
            summary += "Higher novelty was decisive factor. "
        if len(winner.testable_predictions) > len(loser.testable_predictions):
            summary += "More testable predictions provided advantage. "
        return summary


__all__ = ["RankingAgent"]
