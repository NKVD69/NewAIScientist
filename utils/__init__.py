"""
Utility modules for the NewAIScientist system.
"""
from .adjudication import adjudicate, format_verdict_report, parse_measurements
from .llm import ensure_str, get_llm_completion, get_llm_usage_stats, parse_json_response
from .safety import DANGEROUS_MODULES, check_code_safety
from .novelty import NoveltyReport, assess_novelty
from .sandbox_runner import isolation_report, run_sandboxed
from .semantic_scholar import S2Paper, SemanticScholarClient, search_semantic_scholar

__all__ = [
    "get_llm_completion",
    "get_llm_usage_stats",
    "parse_json_response",
    "ensure_str",
    "check_code_safety",
    "DANGEROUS_MODULES",
    # Adjudication (prediction <-> measurement)
    "adjudicate",
    "parse_measurements",
    "format_verdict_report",
    # Sandboxed execution
    "run_sandboxed",
    "isolation_report",
    # Semantic Scholar
    "SemanticScholarClient",
    "S2Paper",
    "search_semantic_scholar",
    # Grounded novelty
    "assess_novelty",
    "NoveltyReport",
]
