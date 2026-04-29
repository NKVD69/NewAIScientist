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
from utils.llm import get_llm_completion, parse_json_response, ensure_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """Tournament-based hypothesis ranking using Elo system with LLM-as-judge."""

    name = "Ranking"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.k_factor = 32  # Elo K-factor
        self.matches_completed = 0
    
    async def conduct_tournament_match(self,
                                      hyp_a: Hypothesis,
                                      hyp_b: Hypothesis) -> Tuple[str, TournamentMatch]:
        """
        Conduct pairwise hypothesis comparison through simulated scientific debate.
        Returns winner ID and match record.
        """
        winner_id = await self._simulate_debate(hyp_a, hyp_b)
        debate_summary = self._generate_debate_summary(hyp_a, hyp_b, winner_id)
        
        # Update Elo ratings
        self._update_elo_ratings(hyp_a, hyp_b, winner_id)
        
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
        Determine the winning hypothesis via LLM-as-judge (primary) or
        score-based heuristic with randomness (fallback).
        """
        if self.llm_client:
            try:
                winner_id = await self._llm_debate(hyp_a, hyp_b)
                if winner_id:
                    return winner_id
            except Exception as e:
                logger.warning("LLM debate failed, falling back to heuristic: %s", e)

        # Fallback: score + noise
        score_a = self._compute_debate_score(hyp_a)
        score_b = self._compute_debate_score(hyp_b)
        debate_factor_a = random.uniform(0.85, 1.15)
        debate_factor_b = random.uniform(0.85, 1.15)
        return hyp_a.id if score_a * debate_factor_a > score_b * debate_factor_b else hyp_b.id

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
    
    def _update_elo_ratings(self, hyp_a: Hypothesis, hyp_b: Hypothesis, winner_id: str):
        expected_a = 1 / (1 + 10 ** ((hyp_b.elo_rating - hyp_a.elo_rating) / 400))
        expected_b = 1 - expected_a
        if winner_id == hyp_a.id:
            hyp_a.elo_rating += self.k_factor * (1 - expected_a)
            hyp_b.elo_rating += self.k_factor * (0 - expected_b)
        else:
            hyp_a.elo_rating += self.k_factor * (0 - expected_a)
            hyp_b.elo_rating += self.k_factor * (1 - expected_b)
    
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
