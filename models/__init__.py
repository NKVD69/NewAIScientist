"""
Data models for the NewAIScientist system.
Re-exports all core dataclasses for convenience.
"""
from .hypothesis import (
    Claim,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    HypothesisLink,
    Prediction,
    ReviewCritique,
    ResearchGoal,
    ResearchQuestion,
    ScoredQuestion,
    StateOfArt,
    StudyPhase,
    UserFeedback,
    VariableRole,
    Variable,
    ExperimentalProtocol,
    AnalysisPlan,
    StatisticalResult,
    DatasetInfo,
    ManuscriptSection,
    Manuscript,
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
    # Memory
    "ContextMemory",
    "TournamentMatch",
]
