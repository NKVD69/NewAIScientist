"""
Data models for the NewAIScientist system.
Re-exports all core dataclasses for convenience.
"""
from .hypothesis import Hypothesis, HypothesisStatus, ReviewCritique, ResearchGoal
from .memory import ContextMemory, TournamentMatch

__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "ReviewCritique",
    "ResearchGoal",
    "ContextMemory",
    "TournamentMatch",
]
