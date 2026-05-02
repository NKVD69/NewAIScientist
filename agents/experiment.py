"""
agents/experiment.py — ExperimentAgent for running code-based experiments.

Responsible for:
- Generating experiment code via LLM
- AST-based safety checking before execution
- Running experiments in isolated subprocess
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from models.hypothesis import Hypothesis, ResearchGoal
from utils.llm import get_llm_completion
from utils.safety import check_code_safety

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ExperimentAgent(BaseAgent):
    """Agent that generates and runs python code to simulate experiments or analyze data"""

    name = "Experiment"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.experiments_run = 0

    async def feasibility_check(
        self,
        hypothesis: Hypothesis,
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict[str, Any]:
        """Estimate sample size and validate biomedical entities mentioned.

        See :mod:`utils.experiment_sandbox` for the underlying machinery.
        Useful as a pre-flight check before running a full LLM-generated
        experiment: a hypothesis whose entities are unresolvable, or whose
        required sample size is impractical, can be flagged early.
        """
        from utils.experiment_sandbox import (
            estimate_required_n,
            feasibility_summary,
            validate_entities,
        )

        text = " ".join([
            hypothesis.title or "",
            hypothesis.description or "",
            hypothesis.mechanism or "",
            " ".join(hypothesis.testable_predictions or []),
        ])
        try:
            entities = await validate_entities(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity validation failed: %s", exc)
            entities = []

        n = estimate_required_n(effect_size, alpha=alpha, power=power)
        summary = feasibility_summary(n, entities)
        logger.info(
            "Feasibility for '%s': n=%s, %d/%d entities verified.",
            (hypothesis.title or "")[:40], n,
            summary["n_entities_verified"], summary["n_entities_total"],
        )
        return summary

    async def run_experiment(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        if not self.llm_client:
            return "Simulation skipped: No LLM available for experimental design."

        logger.info("Designing experiment for hypothesis: %s", hypothesis.title)

        prompt = f"""
        Research Goal: {goal.title}
        Hypothesis: {hypothesis.title}
        Mechanism: {hypothesis.mechanism}
        Predictions: {', '.join(hypothesis.testable_predictions)}

        You are an AI Scientist tasked with empirically validating this hypothesis.
        Write a Python 3 script that uses standard libraries (numpy, scipy, sklearn) to run a simulation or statistical test for the predictions of this hypothesis.
        It MUST be completely self-contained, without assuming external files exist unless you scrape them. Do not use UI libraries.
        It MUST print a clear summary of the results to stdout at the end, concluding whether the data supports the hypothesis.

        Output ONLY the python code inside a ```python block. No other explanation.
        """

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False
            )
            content = response.choices[0].message.content.strip()

            code = ""
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                code = content.split("```")[1].split("```")[0].strip()
            else:
                code = content

            if not code:
                return "Failed to generate experimental code."

            # --- AST Safety Check before execution ---
            is_safe, reason = check_code_safety(code)
            if not is_safe:
                msg = f"Experiment blocked by safety filter: {reason}"
                logger.warning(msg)
                hypothesis.experimental_results = msg
                return msg

            logger.info("Safety check passed. Executing experiment script...")

            import subprocess
            import tempfile

            env = os.environ.copy()
            for key in ('OPENAI_API_KEY', 'NCBI_API_KEY', 'ANTHROPIC_API_KEY'):
                env.pop(key, None)

            # Remove network-related env vars to partially restrict network access
            env.pop('HTTP_PROXY', None)
            env.pop('HTTPS_PROXY', None)

            # Execution in isolated directory
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = os.path.join(temp_dir, 'experiment.py')
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(code)

                # Future extension: if getattr(config, 'USE_DOCKER', False):
                #     write Dockerfile and execute via `docker build` and `docker run --network none`

                try:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, '-S', script_path],
                        cwd=temp_dir,  # Isolate file operations
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env
                    )

                    output = result.stdout
                    if result.stderr:
                        output += f"\nErrors/Warnings:\n{result.stderr}"

                    if not output.strip():
                        output = "Script ran successfully but produced no output."

                    hypothesis.experimental_results = f"Experimental Results:\n{output[:1500]}"
                    self.experiments_run += 1
                    return hypothesis.experimental_results
                except subprocess.TimeoutExpired:
                    hypothesis.experimental_results = "Experiment simulation timed out after 30 seconds."
                    return hypothesis.experimental_results

        except Exception as e:
            hypothesis.experimental_results = f"Experiment implementation failed: {e}"
            return hypothesis.experimental_results


__all__ = ["ExperimentAgent"]
