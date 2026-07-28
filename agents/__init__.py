"""
agents/ package — Specialized AI agents for scientific research.
v3.0 Fully modular agents (extracted from co_scientist.py monolith).
"""

from .analysis import AnalysisAgent
from .base import BaseAgent
from .chaining import HypothesisChainingAgent
from .critic import DevilsAdvocateAgent
from .evolution import EvolutionAgent
from .experiment import ExperimentAgent
from .generation import GenerationAgent
from .graph_agent import GraphAgent
from .literature import LiteratureAgent
from .meta_review import MetaReviewAgent
from .protocol import ProtocolAgent
from .proximity import ProximityAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent
from .preregistration import PreregistrationAgent
from .replication import ReplicationAgent

# v3.0 New Agents
from .scoping import ScopingAgent
from .supervisor import SupervisorAgent, Task
from .utils_agents import SearchAgent
from .writing import WritingAgent

__all__ = [
    "BaseAgent",
    # Core agents (v2.2)
    "LiteratureAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "PreregistrationAgent",
    "ReplicationAgent",
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
    "DevilsAdvocateAgent",
    "HypothesisChainingAgent",
    "SearchAgent"
]
