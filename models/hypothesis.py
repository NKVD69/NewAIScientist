"""
Core hypothesis data structures for the NewAIScientist system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class HypothesisStatus(Enum):
    """Hypothesis lifecycle states"""
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    IN_TOURNAMENT = "in_tournament"
    RANKED = "ranked"
    EVOLVED = "evolved"
    COMPLETED = "completed"


@dataclass
class ReviewCritique:
    """Structure for review feedback"""
    review_type: str  # initial, full, deep_verification, observation, simulation, recurrent
    correctness_score: float  # 0-1
    novelty_score: float      # 0-1
    testability_score: float  # 0-1
    quality_score: float      # 0-1
    feedback: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Hypothesis:
    """Core hypothesis data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    reasoning: str = ""       # Logic/papers that led to this hypothesis
    mechanism: str = ""
    testable_predictions: List[str] = field(default_factory=list)
    grounding_evidence: List[str] = field(default_factory=list)
    experimental_results: str = ""

    # Quality metrics
    elo_rating: float = 1200.0          # Initial Elo rating
    novelty_level: str = "unknown"      # low, medium, high, very_high

    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.GENERATED
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reviews: List[ReviewCritique] = field(default_factory=list)

    # Genealogy
    parent_ids: List[str] = field(default_factory=list)
    generation_method: str = "initial"  # initial, evolved, combined, inspired

    # Citations
    cited_papers: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class ResearchGoal:
    """Research goal specification from scientist"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    domain: str = ""          # biomedicine, physics, chemistry, etc.
    preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
