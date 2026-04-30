"""
Data models for the NewAIScientist system.
Re-exports all core dataclasses for convenience.
"""
from .hypothesis import (
    Hypothesis,
    HypothesisStatus,
    HypothesisLink,
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
    "Hypothesis",
    "HypothesisLink",
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
