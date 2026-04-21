"""
agents/meta_review.py — MetaReviewAgent for system-level synthesis.

Responsible for:
- Identifying top hypotheses and common patterns
- Analyzing tournament results
- Suggesting improvements for next iterations
- Generating research overviews
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.hypothesis import Hypothesis, ResearchGoal
from models.memory import TournamentMatch

logger = logging.getLogger(__name__)


class MetaReviewAgent:
    """Synthesizes insights and provides system-level feedback"""
    
    def __init__(self):
        self.name = "Meta-Review"
        self.meta_reviews_generated = 0
    
    async def generate_meta_review(self, 
                                  hypotheses: List[Hypothesis],
                                  tournament_history: List[TournamentMatch],
                                  goal: ResearchGoal) -> Dict[str, Any]:
        """
        Synthesize insights from reviews and tournaments.
        Identify recurring patterns and improvement opportunities.
        """
        meta_review = {
            "timestamp": datetime.now().isoformat(),
            "total_hypotheses": len(hypotheses),
            "top_hypotheses": self._identify_top_hypotheses(hypotheses),
            "common_strengths": self._identify_common_strengths(hypotheses),
            "common_weaknesses": self._identify_common_weaknesses(hypotheses),
            "tournament_patterns": self._analyze_tournament_patterns(tournament_history),
            "suggested_improvements": self._suggest_improvements(hypotheses),
            "research_overview": self._generate_research_overview(hypotheses, goal),
            "next_iterations_focus": self._suggest_focus_areas(hypotheses)
        }
        
        self.meta_reviews_generated += 1
        return meta_review
    
    def _identify_top_hypotheses(self, hypotheses: List[Hypothesis], 
                                top_k: int = 5) -> List[Dict]:
        sorted_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:top_k]
        return [
            {
                "id": h.id,
                "title": h.title,
                "elo_rating": h.elo_rating,
                "novelty": h.novelty_level,
                "num_reviews": len(h.reviews)
            }
            for h in sorted_hyps
        ]
    
    def _identify_common_strengths(self, hypotheses: List[Hypothesis]) -> List[str]:
        strengths = []
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:5]
        if all(len(h.reviews) > 0 for h in top_hyps):
            strengths.append("Multiple review iterations improve hypothesis quality")
        if all(h.novelty_level in ["high", "very_high"] for h in top_hyps):
            strengths.append("Novelty is a strong factor in ranking")
        if all(len(h.testable_predictions) >= 2 for h in top_hyps):
            strengths.append("Multiple testable predictions increase competitiveness")
        return strengths if strengths else ["Diverse hypothesis portfolio maintained"]
    
    def _identify_common_weaknesses(self, hypotheses: List[Hypothesis]) -> List[str]:
        weaknesses = []
        bottom_hyps = sorted(hypotheses, key=lambda h: h.elo_rating)[:5]
        if any(len(h.reviews) == 0 for h in bottom_hyps):
            weaknesses.append("Unreviewed hypotheses tend to rank lower - prioritize review")
        if any(len(h.testable_predictions) == 0 for h in bottom_hyps):
            weaknesses.append("Lack of testable predictions is a weakness - add empirical angles")
        if any(h.novelty_level == "low" for h in bottom_hyps):
            weaknesses.append("Low novelty is penalized - encourage more creative generation")
        return weaknesses if weaknesses else ["No clear common weaknesses"]
    
    def _analyze_tournament_patterns(self, tournament_history: List[TournamentMatch]) -> Dict:
        if not tournament_history:
            return {"total_matches": 0, "analysis": "No tournaments completed yet"}
        win_counts = defaultdict(int)
        for match in tournament_history:
            win_counts[match.winner_id] += 1
        return {
            "total_matches": len(tournament_history),
            "top_winner": max(win_counts, key=win_counts.get) if win_counts else None,
            "wins_distribution": dict(sorted(win_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _suggest_improvements(self, hypotheses: List[Hypothesis]) -> List[str]:
        suggestions = []
        reviewed_hyps = [h for h in hypotheses if h.reviews]
        if reviewed_hyps:
            avg_novelty_score = sum(
                sum(r.novelty_score for r in h.reviews) / len(h.reviews)
                for h in reviewed_hyps
            ) / len(reviewed_hyps)
            if avg_novelty_score < 0.6:
                suggestions.append("Enhance novelty generation - explore more unconventional directions")
        unreviewed = [h for h in hypotheses if len(h.reviews) == 0]
        if unreviewed:
            suggestions.append(f"Review {len(unreviewed)} unreviewed hypotheses")
        low_testability = [h for h in hypotheses if len(h.testable_predictions) < 2]
        if low_testability:
            suggestions.append(f"Add testable predictions to {len(low_testability)} hypotheses")
        return suggestions if suggestions else ["Continue current trajectory - good progress"]
    
    def _generate_research_overview(self, hypotheses: List[Hypothesis], goal: ResearchGoal) -> str:
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
        overview = f"Research Overview for: {goal.title}\n"
        overview += f"Domain: {goal.domain}\n\n"
        overview += "Top Research Directions:\n"
        for i, hyp in enumerate(top_hyps, 1):
            overview += f"\n{i}. {hyp.title}\n"
            overview += f"   Mechanism: {hyp.mechanism[:100]}...\n"
            overview += f"   Elo Rating: {hyp.elo_rating:.0f}\n"
        return overview
    
    def _suggest_focus_areas(self, hypotheses: List[Hypothesis]) -> List[str]:
        focus_areas = []
        top_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:3]
        focus_areas.append(f"Evolve top 3 hypotheses: {', '.join(h.title[:30] for h in top_hyps)}")
        unreviewed = [h for h in hypotheses if len(h.reviews) == 0]
        if unreviewed:
            focus_areas.append(f"Complete reviews for {len(unreviewed)} hypotheses")
        focus_areas.append("Conduct tournament matches among top performers")
        return focus_areas


__all__ = ["MetaReviewAgent"]
