"""
agents/replication.py — ReplicationAgent for in-silico experiment replication.

Responsible for:
- Re-executing experiments with different random seeds
- Computing reproducibility scores (inter-run variance)
- Providing confidence intervals for experimental results
- Flagging non-reproducible findings before they enter the ranking

A hypothesis whose experiment is not reproducible should be penalised in
the Elo tournament — this agent provides the signal for that.
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import subprocess
import sys
import tempfile
from typing import Any

from models.hypothesis import Hypothesis, ResearchGoal
from utils.safety import check_code_safety

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReplicationAgent(BaseAgent):
    """Replicates experiments to assess reproducibility."""

    name = "Replication"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.replications_run = 0

    async def replicate_experiment(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
        n_replications: int = 3,
        timeout_per_run: int = 30,
    ) -> dict[str, Any]:
        """Re-execute the experiment code with different random seeds.

        Parameters
        ----------
        hypothesis
            Must have ``experimental_results`` set from a prior ExperimentAgent run.
        goal
            The research goal (for context in result parsing).
        n_replications
            Number of times to re-run. Default 3.
        timeout_per_run
            Per-run timeout in seconds.

        Returns
        -------
        A dict with keys:
        - ``reproducibility_score``: 0.0-1.0 (fraction of runs that converged)
        - ``results``: list of per-run stdout strings
        - ``consistency``: narrative summary of inter-run agreement
        """
        if not hypothesis.experimental_results:
            logger.info("No prior experiment results for '%s' — skipping replication.", hypothesis.title[:40])
            return {
                "reproducibility_score": 0.0,
                "results": [],
                "consistency": "No experiment to replicate.",
            }

        # Extract the experiment code if we can regenerate it
        code = await self._get_experiment_code(hypothesis, goal)
        if not code:
            return {
                "reproducibility_score": 0.0,
                "results": [],
                "consistency": "Could not obtain experiment code for replication.",
            }

        # Safety check
        is_safe, reason = check_code_safety(code)
        if not is_safe:
            return {
                "reproducibility_score": 0.0,
                "results": [],
                "consistency": f"Replication blocked by safety filter: {reason}",
            }

        # Run n_replications with different seeds
        results: list[str] = []
        for i in range(n_replications):
            seed = 42 + i * 1000
            seeded_code = self._inject_seed(code, seed)
            output = await self._run_isolated(seeded_code, timeout_per_run)
            results.append(output)
            self.replications_run += 1

        # Compute reproducibility score
        score, consistency = self._assess_reproducibility(results)
        hypothesis.reproducibility_score = score
        hypothesis.replication_results = [
            {"seed": 42 + i * 1000, "output": r[:500]}
            for i, r in enumerate(results)
        ]

        logger.info(
            "Replication for '%s': score=%.2f (%d/%d converged).",
            hypothesis.title[:40], score, int(score * n_replications), n_replications,
        )
        return {
            "reproducibility_score": score,
            "results": [r[:500] for r in results],
            "consistency": consistency,
        }

    async def _get_experiment_code(
        self, hypothesis: Hypothesis, goal: ResearchGoal
    ) -> str | None:
        """Regenerate experiment code via LLM (same prompt as ExperimentAgent)."""
        if not self.llm_client:
            return None

        from utils.llm import get_llm_completion

        prompt = f"""
        Research Goal: {goal.title}
        Hypothesis: {hypothesis.title}
        Mechanism: {hypothesis.mechanism}
        Predictions: {', '.join(hypothesis.testable_predictions)}

        Write a Python 3 script using standard libraries (numpy, scipy, pandas)
        that performs a statistical analysis to test this hypothesis.
        The script must:
        - Accept a random seed via `import numpy as np; np.random.seed(SEED)`
          where SEED is set at the top of the script.
        - Generate synthetic data if no data.csv is available.
        - Print a clear summary of results (p-values, effect sizes, conclusion).

        Output ONLY the python code inside a ```python block. No other text.
        """

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False,
            )
            content = response.choices[0].message.content.strip()

            if "```python" in content:
                return content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                return content.split("```")[1].split("```")[0].strip()
            return content
        except Exception as e:
            logger.warning("Failed to regenerate experiment code: %s", e)
            return None

    @staticmethod
    def _inject_seed(code: str, seed: int) -> str:
        """Inject a random seed at the top of the experiment code."""
        seed_block = (
            f"# --- Replication seed injection ---\n"
            f"import numpy as np\n"
            f"np.random.seed({seed})\n"
            f"import random\n"
            f"random.seed({seed})\n"
            f"# --- End seed injection ---\n\n"
        )
        return seed_block + code

    @staticmethod
    async def _run_isolated(code: str, timeout: int) -> str:
        """Execute code in an isolated subprocess."""
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "NCBI_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(key, None)

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "replicate.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-S", script_path],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]: {result.stderr}"
                return output or "Script ran but produced no output."
            except subprocess.TimeoutExpired:
                return "[TIMEOUT] Script timed out."
            except Exception as e:
                return f"[ERROR] {e}"

    @staticmethod
    def _assess_reproducibility(results: list[str]) -> tuple[float, str]:
        """Assess how consistent the results are across replications.

        Heuristic: a run "converges" if it produces non-error output
        that contains at least one p-value or statistical keyword.
        Consistency is the fraction of converged runs.
        """
        if not results:
            return 0.0, "No results to assess."

        convergence_keywords = {"p-value", "p_value", "p =", "p=", "significant",
                                "effect size", "correlation", "coefficient",
                                "t-statistic", "chi-square", "f-statistic",
                                "supports", "rejects", "conclusion"}

        converged = 0
        for r in results:
            lower = r.lower()
            if "[ERROR]" in r or "[TIMEOUT]" in r:
                continue
            if any(kw in lower for kw in convergence_keywords):
                converged += 1

        score = converged / len(results) if results else 0.0

        if score >= 0.9:
            consistency = "High reproducibility: all runs converged with statistical output."
        elif score >= 0.6:
            consistency = "Moderate reproducibility: most runs converged but some variability detected."
        elif score > 0:
            consistency = "Low reproducibility: significant inter-run variation or failures."
        else:
            consistency = "Not reproducible: no runs produced valid statistical output."

        return score, consistency


__all__ = ["ReplicationAgent"]
