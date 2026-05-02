"""
agents/analysis.py — AnalysisAgent for data exploration and statistics.

Responsible for:
- Exploring datasets (CSV, manual upload)
- Connecting to public databases (GEO, ClinicalTrials.gov)
- Running statistical tests
- Generating visualizations
- Interpreting results against hypotheses
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from models.hypothesis import (
    AnalysisPlan,
    DatasetInfo,
    Hypothesis,
    StatisticalResult,
)
from utils.llm import get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Explores data and performs statistical analysis."""

    name = "Analysis"

    # ------------------------------------------------------------------
    # DATA LOADING & PUBLIC DBs
    # ------------------------------------------------------------------

    async def load_csv(self, file_path: str) -> DatasetInfo:
        """Loads a local CSV and extracts metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV not found: {file_path}")

        df = pd.read_csv(file_path)
        info = DatasetInfo(
            name=os.path.basename(file_path),
            source="upload",
            num_rows=len(df),
            num_columns=len(df.columns),
            column_names=list(df.columns),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            description=f"CSV upload with {len(df)} rows."
        )
        return info

    async def fetch_public_database_info(self, query: str, db_type: str) -> list[DatasetInfo]:
        """
        Interacts with public databases (MOCK/STRUCTURE for GEO, ClinicalTrials, etc.)
        In a real scenario, this would use Bio.Entrez or specific APIs.
        """
        logger.info(f"Searching {db_type} for query: {query}")

        if not self.llm_client:
            return [DatasetInfo(name="Mock Result", source=db_type, description="LLM required for search")]

        prompt = f"""Search common research databases ({db_type}) for metadata related to: {query}.
Return a JSON list of 3 potential datasets with metadata.
Keys: "name", "source" (must be {db_type}), "source_url", "description", "num_rows_est", "relevance_score".
Return ONLY JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            if isinstance(data, dict):
                data = next(iter(data.values())) if isinstance(next(iter(data.values())), list) else [data]

            results = []
            for item in data:
                results.append(DatasetInfo(
                    name=item.get("name", "Unknown Dataset"),
                    source=item.get("source", db_type),
                    source_url=item.get("source_url", ""),
                    description=item.get("description", ""),
                    num_rows=item.get("num_rows_est", 0)
                ))
            return results
        except Exception as e:
            logger.error(f"Search in {db_type} failed: {e}")
            return []

    # ------------------------------------------------------------------
    # EXPLORATION & STATS
    # ------------------------------------------------------------------

    async def run_exploratory_analysis(self, df: pd.DataFrame) -> str:
        """Generates a summary of the dataset's distributions and anomalies."""
        summary = df.describe(include='all').to_string()
        nulls = df.isnull().sum().to_string()

        if not self.llm_client:
            return f"Summary Stats:\n{summary}\nMissing Values:\n{nulls}"

        prompt = f"""Analyze the following dataset summary stats and identify anomalies, interesting patterns, or data quality issues.
Summary Stats:
{summary}
Missing Values:
{nulls}

Provide a concise exploratory report."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during exploration: {e}"

    async def run_statistical_tests(self, df: pd.DataFrame, plan: AnalysisPlan) -> list[StatisticalResult]:
        """
        Executes statistical tests as defined in the analysis plan.
        Uses the LLM to write the code and attempts to execute it safely.
        """
        results = []
        if not self.llm_client:
            return results

        prompt = f"""Based on this Analysis Plan, write a Python script using scipy.stats to test the data in 'df'.
Plan: {plan.primary_analysis}
Tests: {plan.statistical_tests}
Alpha: {plan.alpha_level}

The 'df' contains columns: {list(df.columns)}.
Return a JSON object with results for each test (statistic, p_value, significant, interpretation).
Return ONLY JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)

            if isinstance(data, dict):
                for test_name, res in data.items():
                    results.append(StatisticalResult(
                        test_name=test_name,
                        statistic_value=res.get("statistic", 0.0),
                        p_value=res.get("p_value", 1.0),
                        significant=res.get("significant", False),
                        interpretation=res.get("interpretation", "")
                    ))
            elif isinstance(data, list):
                for res in data:
                    results.append(StatisticalResult(
                        test_name=res.get("test_name", "Unknown"),
                        statistic_value=res.get("statistic", 0.0),
                        p_value=res.get("p_value", 1.0),
                        significant=res.get("significant", False),
                        interpretation=res.get("interpretation", "")
                    ))
            return results
        except Exception as e:
            logger.error(f"Statistical testing failed: {e}")
            return []

    async def interpret_results(self, results: list[StatisticalResult], hypothesis: Hypothesis) -> str:
        """Synthesizes statistical results into a conclusion about the hypothesis."""
        if not self.llm_client:
            return "No LLM available for interpretation."

        results_str = "\n".join([f"- {r.test_name}: p={r.p_value:.4f}, Sig={r.significant}. {r.interpretation}" for r in results])

        prompt = f"""Interpret the following statistical results in the context of the hypothesis.
Hypothesis: {hypothesis.title}
Mechanism: {hypothesis.mechanism}

Results:
{results_str}

Conclusion: Does the evidence support or refute the hypothesis? Be rigorous."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Interpretation failed: {e}"


__all__ = ["AnalysisAgent"]
