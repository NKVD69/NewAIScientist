"""
Persistent memory structures for the NewAIScientist system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from .hypothesis import Hypothesis, ResearchGoal


@dataclass
class TournamentMatch:
    """Record of a pairwise hypothesis comparison"""
    match_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_a_id: str = ""
    hypothesis_b_id: str = ""
    winner_id: str = ""
    debate_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ContextMemory:
    """
    Persistent memory of the co-scientist system state.
    Holds all hypotheses, literature, tournament records and meta-reviews
    generated over a research session.
    """
    research_goal: ResearchGoal = field(default_factory=ResearchGoal)
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    tournament_history: List[TournamentMatch] = field(default_factory=list)
    agent_performance_stats: Dict[str, Dict] = field(default_factory=dict)
    iteration_count: int = 0
    literature_context: List[Dict] = field(default_factory=list)   # Retrieved papers
    meta_reviews: List[Dict] = field(default_factory=list)          # Meta-review results
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
