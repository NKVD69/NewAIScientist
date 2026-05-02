"""
Core data structures for the NewAIScientist system.

Contains all dataclasses used across agents, orchestrator, and API:
- Enums: HypothesisStatus, StudyPhase, VariableRole
- Research: ResearchGoal, ResearchQuestion, StateOfArt, ScoredQuestion
- Hypothesis: Hypothesis, ReviewCritique, HypothesisLink
- Experimental: Variable, ExperimentalProtocol, AnalysisPlan
- Analysis: StatisticalResult, DatasetInfo
- Writing: ManuscriptSection, Manuscript
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# ============================================================================
# ENUMS
# ============================================================================

class HypothesisStatus(Enum):
    """Hypothesis lifecycle states"""
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    IN_TOURNAMENT = "in_tournament"
    RANKED = "ranked"
    EVOLVED = "evolved"
    COMPLETED = "completed"


class StudyPhase(Enum):
    """Phase of the scientific workflow"""
    SCOPING = "scoping"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENTAL_DESIGN = "experimental_design"
    DATA_ANALYSIS = "data_analysis"
    WRITING = "writing"


class VariableRole(Enum):
    """Role of a variable in an experimental design"""
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    CONTROL = "control"
    CONFOUNDING = "confounding"


# ============================================================================
# CLAIM / EVIDENCE / CONFIDENCE
# ============================================================================

@dataclass
class Evidence:
    """A single piece of evidence supporting (or contradicting) a claim.

    ``source_type`` is one of: ``"rag"`` (RAG chunk), ``"citation"``
    (external DOI/arXiv/PMID), ``"prior"`` (general prior knowledge),
    ``"experiment"`` (data we generated ourselves).
    ``polarity`` is +1 if the evidence supports the claim, -1 if it
    contradicts it, 0 if it is purely contextual.
    """
    text: str = ""
    source_type: str = "prior"
    source_ref: str = ""           # DOI / arXiv ID / chunk ID / dataset ID
    polarity: int = 1
    confidence: float = 0.5        # 0..1, the agent's belief in this evidence

    def __post_init__(self):
        if self.polarity not in (-1, 0, 1):
            raise ValueError(f"polarity must be -1, 0, or 1; got {self.polarity}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )


@dataclass
class Claim:
    """An atomic, falsifiable assertion making up a hypothesis.

    ``confidence`` is the agent's belief in this specific claim, separate
    from the claim's evidentiary weight (which is computed from the
    aggregated polarity*confidence of its ``evidence`` list).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )

    def evidence_score(self) -> float:
        """Aggregate score in [-1, 1]: weighted sum of polarities × confidences."""
        if not self.evidence:
            return 0.0
        total = sum(abs(e.confidence) for e in self.evidence) or 1.0
        return sum(e.polarity * e.confidence for e in self.evidence) / total


# ============================================================================
# FALSIFIABLE PREDICTIONS
# ============================================================================

@dataclass
class Prediction:
    """A quantitative, pre-registered prediction tied to a claim.

    A hypothesis with no ``Prediction`` instances is by construction
    not falsifiable. The ``refuting_threshold`` makes refutation explicit:
    if the measured ``quantity`` falls outside ``[expected_value - ci,
    expected_value + ci]`` AND beyond ``refuting_threshold``, the parent
    claim is considered refuted.
    """
    quantity: str = ""
    expected_value: float = 0.0
    ci: float = 0.0                # symmetric ±CI around expected_value
    unit: str = ""
    refuting_threshold: float = 0.0
    rationale: str = ""

    def is_falsifiable(self) -> bool:
        """A prediction is falsifiable iff the refuting threshold is non-trivial."""
        return self.refuting_threshold > 0 and self.ci >= 0

    def is_refuted_by(self, observed: float) -> bool:
        """Return True if ``observed`` falls beyond the refuting threshold."""
        if not self.is_falsifiable():
            return False
        return abs(observed - self.expected_value) > self.refuting_threshold


# ============================================================================
# REVIEW & CRITIQUE
# ============================================================================

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


# ============================================================================
# HYPOTHESIS
# ============================================================================

@dataclass
class Hypothesis:
    """Core hypothesis data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    reasoning: str = ""       # Logic/papers that led to this hypothesis
    mechanism: str = ""
    testable_predictions: list[str] = field(default_factory=list)
    grounding_evidence: list[str] = field(default_factory=list)
    experimental_results: str = ""

    # Quality metrics
    elo_rating: float = 1200.0          # Initial Elo rating
    novelty_level: str = "unknown"      # low, medium, high, very_high

    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.GENERATED
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reviews: list[ReviewCritique] = field(default_factory=list)

    # Genealogy
    parent_ids: list[str] = field(default_factory=list)
    generation_method: str = "initial"  # initial, evolved, combined, inspired

    # Citations
    cited_papers: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # v3.0 — Hypothesis chaining (Sprint 3)
    linked_hypotheses: list[tuple[str, str]] = field(default_factory=list)
    # Each tuple: (target_hypothesis_id, link_type)
    # link_type: "supports", "contradicts", "depends_on"

    # Structured decomposition (improvement #2)
    claims: list["Claim"] = field(default_factory=list)
    # Falsifiable, pre-registered predictions (improvement #3)
    falsifiable_predictions: list["Prediction"] = field(default_factory=list)
    # Falsifiability score (0..1), filled by FalsifiabilityAgent
    falsifiability_score: float = 0.0


@dataclass
class UserFeedback:
    """A scientist's structured feedback on a single hypothesis.

    Used by the interactive feedback loop: the scientist either accepts,
    rejects, or asks for a refinement of an evolved hypothesis. The free-text
    ``comment`` is fed back into ``EvolutionAgent.evolve_with_feedback`` so
    the next iteration can incorporate the critique.
    """
    hypothesis_id: str = ""
    verdict: str = "refine"  # one of: "agree", "disagree", "refine"
    comment: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.verdict not in {"agree", "disagree", "refine"}:
            raise ValueError(
                f"verdict must be 'agree' | 'disagree' | 'refine', got {self.verdict!r}"
            )


@dataclass
class HypothesisLink:
    """Directed link between two hypotheses"""
    source_id: str = ""
    target_id: str = ""
    link_type: str = "supports"  # supports, contradicts, depends_on
    reasoning: str = ""


# ============================================================================
# RESEARCH GOAL & QUESTIONS
# ============================================================================

@dataclass
class ResearchGoal:
    """Research goal specification from scientist"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    domain: str = ""          # biomedicine, physics, chemistry, etc.
    preferences: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResearchQuestion:
    """Structured research question (PICO/FINER format)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    type: str = "exploratory"  # descriptive, correlational, causal, exploratory
    pico: Optional[dict] = None  # Population, Intervention, Comparison, Outcome
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    parent_gap: str = ""  # The gap in the state of the art that motivates this question


@dataclass
class ScoredQuestion:
    """Research question with composite relevance score"""
    question: ResearchQuestion = field(default_factory=ResearchQuestion)
    composite_score: float = 0.0  # Weighted combination of novelty, feasibility, impact


@dataclass
class StateOfArt:
    """Structured synthesis of current knowledge"""
    known_facts: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    summary: str = ""


# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

@dataclass
class Variable:
    """Variable in an experimental design"""
    name: str = ""
    role: str = "independent"  # independent, dependent, control, confounding
    description: str = ""
    measurement_method: str = ""
    data_type: str = "continuous"  # continuous, categorical, ordinal, binary
    unit: str = ""
    expected_range: str = ""


@dataclass
class ExperimentalProtocol:
    """Structured experimental protocol"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    title: str = ""
    design_type: str = "simulation"  # RCT, quasi-experimental, observational, simulation
    variables: list[Variable] = field(default_factory=list)
    experimental_groups: list[str] = field(default_factory=list)
    control_group: str = ""
    randomization_method: str = ""
    blinding: str = "none"  # none, single, double
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    procedure_steps: list[str] = field(default_factory=list)
    sample_size: int = 0
    statistical_tests: list[str] = field(default_factory=list)
    alpha_level: float = 0.05
    corrections: str = "none"  # bonferroni, holm, fdr, none
    code: str = ""  # Executable Python script
    power_analysis: dict = field(default_factory=dict)


@dataclass
class AnalysisPlan:
    """Pre-registered statistical analysis plan"""
    primary_analysis: str = ""
    statistical_tests: list[str] = field(default_factory=list)
    alpha_level: float = 0.05
    corrections: str = "none"


# ============================================================================
# DATA & STATISTICS
# ============================================================================

@dataclass
class StatisticalResult:
    """Result of a single statistical test"""
    test_name: str = ""
    statistic_value: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    interpretation: str = ""


@dataclass
class DatasetInfo:
    """Metadata about a dataset"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    source: str = ""  # upload, GEO, ClinicalTrials, etc.
    source_url: str = ""
    description: str = ""
    num_rows: int = 0
    num_columns: int = 0
    column_names: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)


# ============================================================================
# MANUSCRIPT
# ============================================================================

@dataclass
class ManuscriptSection:
    """A single section of a scientific paper"""
    section_type: str = ""  # abstract, introduction, methods, results, discussion, conclusion
    title: str = ""
    content: str = ""


@dataclass
class Manuscript:
    """Complete scientific manuscript"""
    title: str = ""
    sections: dict[str, ManuscriptSection] = field(default_factory=dict)
    references: list[dict[str, str]] = field(default_factory=list)
