"""
agents/protocol.py — ProtocolAgent for formal experimental design.

Responsible for:
- Designing structured experiments (IV, DV, CV, Groups, Controls)
- Performing statistical power analysis to estimate required sample size
- Generating pre-registered analysis plans
- Creating executable Python scripts for the protocol
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from models.hypothesis import (
    Hypothesis,
    ResearchGoal,
    ExperimentalProtocol,
    Variable,
    VariableRole,
)
from utils.llm import get_llm_completion, parse_json_response, ensure_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ProtocolAgent(BaseAgent):
    """Designs formal experimental protocols and analysis plans."""

    name = "Protocol"

    async def design_experiment(
        self, hypothesis: Hypothesis, goal: ResearchGoal
    ) -> ExperimentalProtocol:
        """
        Creates a structured experimental design for a given hypothesis.
        """
        if not self.llm_client:
            return self._fallback_protocol(hypothesis, goal)

        prompt = f"""You are an expert in experimental design and research methodology.

Research Goal: {goal.title}
Hypothesis: {hypothesis.title}
Mechanism: {hypothesis.mechanism}
Predictions: {", ".join(hypothesis.testable_predictions)}

Design a rigorous experimental protocol to test this hypothesis. 
Return a JSON object with EXACTLY these keys:
{{
  "title": "Protocol Title",
  "design_type": "RCT|quasi-experimental|observational|simulation",
  "variables": [
    {{"name": "VarName", "role": "independent|dependent|control|confounding", "description": "...", "measurement_method": "...", "data_type": "continuous|categorical|ordinal|binary", "unit": "...", "expected_range": "..."}}
  ],
  "experimental_groups": ["Group A", "Group B"],
  "control_group": "Control Group Name",
  "randomization_method": "Description of randomization",
  "blinding": "none|single|double",
  "inclusion_criteria": ["criteria 1"],
  "exclusion_criteria": ["criteria 1"],
  "procedure_steps": ["step 1", "step 2"],
  "statistical_tests": ["t-test", "ANOVA", "etc."],
  "alpha_level": 0.05,
  "corrections": "bonferroni|holm|fdr|none"
}}
Return ONLY the JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            
            variables = []
            for v in data.get("variables", []):
                variables.append(Variable(
                    name=v.get("name", ""),
                    role=v.get("role", "independent"),
                    description=v.get("description", ""),
                    measurement_method=v.get("measurement_method", ""),
                    data_type=v.get("data_type", "continuous"),
                    unit=v.get("unit", ""),
                    expected_range=v.get("expected_range", "")
                ))

            return ExperimentalProtocol(
                hypothesis_id=hypothesis.id,
                title=data.get("title", f"Protocol for {hypothesis.title}"),
                design_type=data.get("design_type", "simulation"),
                variables=variables,
                experimental_groups=data.get("experimental_groups", []),
                control_group=data.get("control_group", ""),
                randomization_method=data.get("randomization_method", ""),
                blinding=data.get("blinding", "none"),
                inclusion_criteria=data.get("inclusion_criteria", []),
                exclusion_criteria=data.get("exclusion_criteria", []),
                procedure_steps=data.get("procedure_steps", []),
                statistical_tests=data.get("statistical_tests", []),
                alpha_level=float(data.get("alpha_level", 0.05)),
                corrections=data.get("corrections", "none")
            )
        except Exception as e:
            logger.error("Experimental design failed: %s", e)
            return self._fallback_protocol(hypothesis, goal)

    async def power_analysis(self, protocol: ExperimentalProtocol) -> Dict[str, Any]:
        """
        Estimates required sample size based on the design.
        """
        if not self.llm_client:
            return {"sample_size": 100, "rationale": "Fallback value"}

        prompt = f"""Perform a statistical power analysis for the following experimental design.

Protocol: {protocol.title}
Design Type: {protocol.design_type}
Statistical Tests: {", ".join(protocol.statistical_tests)}
Alpha Level: {protocol.alpha_level}

Estimate the required sample size (N) to achieve 80% power (beta=0.2) assuming a medium effect size.
Return a JSON object:
{{
  "estimated_sample_size": 150,
  "rationale": "Explanation of the analysis and effect size assumptions",
  "recommended_n_per_group": 75,
  "potential_attrition_rate": 0.1
}}
Return ONLY JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            protocol.sample_size = data.get("estimated_sample_size", 0)
            protocol.power_analysis = data
            return data
        except Exception as e:
            logger.error("Power analysis failed: %s", e)
            return {"sample_size": 100, "rationale": "Error during calculation"}

    async def generate_executable_code(self, protocol: ExperimentalProtocol) -> str:
        """
        Generates a Python script that implements the experimental data collection or simulation.
        """
        if not self.llm_client:
            return "# Code generation requires LLM"

        prompt = f"""Write a Python script that implements the following experimental protocol.
If it's a simulation, generate the synthetic data. 
If it's for analysis, write the code that would process a CSV named 'data.csv'.

Protocol: {protocol.title}
Variables: {[{v.name: v.role} for v in protocol.variables]}
Sample Size: {protocol.sample_size}

The script should:
1. Use numpy, pandas, and scipy.stats.
2. Generate synthetic data if design_type is 'simulation'.
3. Perform the requested statistical tests: {protocol.statistical_tests}.
4. Output results in a clear format.

Return ONLY the code block."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False,
            )
            code = response.choices[0].message.content
            # Clean markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            protocol.code = code
            return code
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            return f"# Failure: {e}"

    def _fallback_protocol(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ExperimentalProtocol:
        return ExperimentalProtocol(
            hypothesis_id=hypothesis.id,
            title=f"Preliminary Protocol for {hypothesis.title}",
            procedure_steps=["Initial observation", "Data collection", "Verification"],
            sample_size=50
        )


__all__ = ["ProtocolAgent"]
