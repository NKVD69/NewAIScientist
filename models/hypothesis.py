from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class HypothesisStatus(Enum):
    GENERATED = "generated"
    REFLECTED = "reflected"
    REVIEWED = "reviewed"
    RANKED = "ranked"
    EVOLVED = "evolved"
    VALIDATED = "validated"
    REJECTED = "rejected"

@dataclass
class Hypothesis:
    """Core hypothesis data model for NewAIScientist v3.0"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    mechanism: str = ""       # The "How" part of the hypothesis
    novelty_level: str = "Medium"
    elo_rating: float = 1200.0
    
    # Scientific components
    testable_predictions: List[str] = field(default_factory=list)
    grounding_evidence: List[Dict] = field(default_factory=list)
    cited_papers: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.GENERATED
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reviews: List[Dict] = field(default_factory=list)
    adversarial_review: Optional[Dict] = None
    
    # Genealogy & Chaining
    parent_id: Optional[str] = None
    link_type: Optional[str] = None  # refines, refutes, supports
    generation_method: str = "initial"
    history: List[Dict] = field(default_factory=list)
    
    # Human Collaboration (Sprint 3)
    scientist_notes: str = ""
    human_feedback: List[Dict] = field(default_factory=list)
    
    metadata: Dict = field(default_factory=dict)

@dataclass
class ResearchGoal:
    """Research goal specification from scientist"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    domain: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ExperimentalProtocol:
    """Formal experimental design"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    title: str = ""
    independent_variables: List[Dict] = field(default_factory=list)
    dependent_variables: List[Dict] = field(default_factory=list)
    control_variables: List[Dict] = field(default_factory=list)
    experimental_groups: List[str] = field(default_factory=list)
    control_group: str = ""
    sample_size: int = 0
    power_analysis: Dict = field(default_factory=dict)
    code: str = ""  # Executable validation code

@dataclass
class StatisticalResult:
    """Results from data analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    protocol_id: str = ""
    results: List[Dict] = field(default_factory=list)
    interpretation: str = ""
    visualizations: List[str] = field(default_factory=list)

@dataclass
class Manuscript:
    """Scientific manuscript draft"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    abstract: str = ""
    sections: Dict[str, Dict] = field(default_factory=dict)
    references: List[Dict] = field(default_factory=list)
    status: str = "draft"
