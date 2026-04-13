"""
agents/ package — Specialized AI agents for scientific research.
v3.0 Extended Agents.
"""

from .literature import LiteratureAgent
from .generation import GenerationAgent
from .reflection import ReflectionAgent
from .ranking import RankingAgent
from .proximity import ProximityAgent
from .evolution import EvolutionAgent
from .meta_review import MetaReviewAgent
from .graph_agent import GraphAgent
# v3.0 New Agents
from .scoping import ScopingAgent
from .protocol import ProtocolAgent
from .analysis import AnalysisAgent
from .writing import WritingAgent

__all__ = [
    "LiteratureAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "ProximityAgent",
    "EvolutionAgent",
    "MetaReviewAgent",
    "GraphAgent",
    "ScopingAgent",
    "ProtocolAgent",
    "AnalysisAgent",
    "WritingAgent",
]
