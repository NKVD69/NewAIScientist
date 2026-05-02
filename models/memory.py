"""
Persistent memory structures for the NewAIScientist system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    # --- Core (v2.2) ---
    research_goal: ResearchGoal = field(default_factory=ResearchGoal)
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    tournament_history: List[TournamentMatch] = field(default_factory=list)
    agent_performance_stats: Dict[str, Dict] = field(default_factory=dict)
    iteration_count: int = 0
    literature_context: List[Dict] = field(default_factory=list)   # Retrieved papers
    meta_reviews: List[Dict] = field(default_factory=list)          # Meta-review results
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    # Per-source ISO timestamp of the last successful literature search.
    # Used by ``LiteratureAgent.refresh()`` to fetch only the delta.
    literature_last_seen: Dict[str, str] = field(default_factory=dict)

    # --- v3.0 Phase tracking ---
    current_phase: str = "scoping"

    # Phase 1: Scoping
    state_of_art: Dict = field(default_factory=dict)
    research_questions: List = field(default_factory=list)
    conceptual_framework: Dict = field(default_factory=dict)

    # Phase 4: Experimental Design
    experimental_protocols: Dict[str, Any] = field(default_factory=dict)

    # Phase 5: Data Analysis
    datasets: Dict[str, Any] = field(default_factory=dict)
    statistical_results: List = field(default_factory=list)
    interpretations: Dict[str, str] = field(default_factory=dict)

    # Phase 6: Writing
    manuscript: Any = None
    manuscript_sections: Dict[str, str] = field(default_factory=dict)
