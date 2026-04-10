"""
agents/ package — NewAIScientist multi-agent system.
Re-exports all agent classes for convenience.
"""
from .literature import LiteratureAgent
from .generation import GenerationAgent
from .reflection import ReflectionAgent
from .ranking import RankingAgent
from .proximity import ProximityAgent
from .evolution import EvolutionAgent
from .experiment import ExperimentAgent
from .graph import GraphAgent
from .meta_review import MetaReviewAgent
from .supervisor import SupervisorAgent, CoScientist

__all__ = [
    "LiteratureAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "ProximityAgent",
    "EvolutionAgent",
    "ExperimentAgent",
    "GraphAgent",
    "MetaReviewAgent",
    "SupervisorAgent",
    "CoScientist",
]
