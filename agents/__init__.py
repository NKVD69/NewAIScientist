"""
agents/ package — Specialized AI agents for scientific research.
v3.0 Fully modular agents (extracted from co_scientist.py monolith).
"""

from .literature import LiteratureAgent
from .generation import GenerationAgent
from .reflection import ReflectionAgent
from .ranking import RankingAgent
from .proximity import ProximityAgent
from .evolution import EvolutionAgent
from .meta_review import MetaReviewAgent
from .graph_agent import GraphAgent
from .experiment import ExperimentAgent
from .supervisor import SupervisorAgent, Task

# v3.0 New Agents
from .scoping import ScopingAgent
from .protocol import ProtocolAgent
from .analysis import AnalysisAgent
from .writing import WritingAgent

__all__ = [
    # Core agents (v2.2)
    "LiteratureAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "ProximityAgent",
    "EvolutionAgent",
    "MetaReviewAgent",
    "GraphAgent",
    "ExperimentAgent",
    "SupervisorAgent",
    "Task",
    # v3.0 New Agents
    "ScopingAgent",
    "ProtocolAgent",
    "AnalysisAgent",
    "WritingAgent",
]
