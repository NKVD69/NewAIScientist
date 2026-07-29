"""
Data models for the NewAIScientist system.
Re-exports all core dataclasses for convenience.
"""
from .hypothesis import (
    AnalysisPlan,
    Claim,
    DatasetInfo,
    Evidence,
    ExperimentalProtocol,
    Hypothesis,
    HypothesisLink,
    HypothesisStatus,
    Manuscript,
    ManuscriptSection,
    Prediction,
    ResearchGoal,
    ResearchQuestion,
    ReviewCritique,
    ScoredQuestion,
    StateOfArt,
    StatisticalResult,
    StudyPhase,
    UserFeedback,
    Variable,
    VariableRole,
)
from .experiment import (
    ExperimentKind,
    ExperimentRun,
    Measurement,
    Verdict,
    VerdictStatus,
)
from .memory import ContextMemory, TournamentMatch

__all__ = [
    # Enums
    "HypothesisStatus",
    "StudyPhase",
    "VariableRole",
    # Research
    "ResearchGoal",
    "ResearchQuestion",
    "ScoredQuestion",
    "StateOfArt",
    # Hypothesis
    "Claim",
    "Evidence",
    "Hypothesis",
    "HypothesisLink",
    "Prediction",
    "ReviewCritique",
    "UserFeedback",
    # Experimental
    "Variable",
    "ExperimentalProtocol",
    "AnalysisPlan",
    # Analysis
    "StatisticalResult",
    "DatasetInfo",
    # Writing
    "ManuscriptSection",
    "Manuscript",
    # Experiment adjudication
    "ExperimentKind",
    "ExperimentRun",
    "Measurement",
    "Verdict",
    "VerdictStatus",
    # Memory
    "ContextMemory",
    "TournamentMatch",
]
