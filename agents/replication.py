"""
agents/replication.py — ReplicationAgent: multiverse (specification-curve) analysis.

Replaces seed-variation "replication", which measured nothing useful:

* On real, deterministic data a different ``np.random.seed`` produces
  **identical** output. The system reported perfect reproducibility — true,
  and entirely uninformative.
* On synthetic data it measured the variance of the pseudo-random generator
  the LLM had just written. "Robust" meant the model picked a small sigma.

The question worth asking is not *does this rerun the same way?* but *does
the conclusion survive the analytic choices the analyst could defensibly
have made otherwise?* That is a specification curve: enumerate the forks
(outlier policy, statistical test, transform, covariate adjustment), run all
of them, report the distribution.

The resulting fragility populates ``Hypothesis.multiverse_fragility``, which
feeds the Bradley-Terry prior — so a finding surviving 4 of 96 defensible
specifications is penalised in the ranking rather than written up as a result.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from models.hypothesis import Hypothesis, ResearchGoal
from utils.multiverse import (
    DEFAULT_FORKS,
    MultiverseReport,
    SpecificationResult,
    build_specification_code,
    enumerate_specifications,
    parse_specification_result,
)
from utils.safety import check_code_safety
from utils.sandbox_runner import SandboxPolicy, run_sandboxed

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReplicationAgent(BaseAgent):
    """Assesses robustness by running the analysis across the fork space."""

    name = "Replication"

    def __init__(
        self,
        use_local_llm: bool = True,
        forks: dict[str, list[str]] | None = None,
        max_specifications: int = 32,
        max_parallel: int = 4,
        policy: SandboxPolicy | None = None,
    ):
        super().__init__(use_local_llm=use_local_llm)
        self.replications_run = 0
        self.forks = forks or DEFAULT_FORKS
        #: The full default space is 96 specifications; sampling keeps the
        #: cost bounded while preserving the distributional picture.
        self.max_specifications = max_specifications
        self.max_parallel = max_parallel
        self.policy = policy or SandboxPolicy.from_env()

    # ------------------------------------------------------------------

    async def replicate_experiment(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
        n_replications: int | None = None,
        timeout_per_run: int = 30,
    ) -> dict[str, Any]:
        """Run the multiverse analysis for one hypothesis.

        ``n_replications`` is accepted for backwards compatibility and, when
        given, caps the number of specifications sampled.

        Returns a dict retaining the legacy keys (``reproducibility_score``,
        ``results``, ``consistency``) plus the full ``multiverse`` report.
        """
        base_code = await self._get_experiment_code(hypothesis, goal)
        if not base_code:
            logger.info(
                "No experiment code available for '%s' — skipping multiverse.",
                (hypothesis.title or "")[:40],
            )
            return self._empty_result("no prior experiment code to vary")

        is_clean, reason = check_code_safety(base_code)
        if not is_clean:
            return self._empty_result(f"code rejected by quality filter: {reason}")

        specs = enumerate_specifications(self.forks)
        cap = min(self.max_specifications, n_replications or self.max_specifications)
        if len(specs) > cap:
            # Deterministic sample so a run is reproducible; the fork space is
            # a grid, so a random subset preserves marginal coverage well.
            specs = random.Random(0).sample(specs, cap)

        logger.info(
            "Multiverse for '%s': %d specifications (of %d in the full grid).",
            (hypothesis.title or "")[:40], len(specs),
            len(enumerate_specifications(self.forks)),
        )

        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _run(spec: dict[str, str]) -> SpecificationResult:
            async with semaphore:
                code = build_specification_code(base_code, spec)
                result = await run_sandboxed(code, policy=self.policy)
                if result.blocked:
                    return SpecificationResult(spec=spec, error=f"blocked: {result.error[:80]}")
                if result.timed_out:
                    return SpecificationResult(spec=spec, error="timed out")
                return parse_specification_result(result.stdout, spec)

        outcomes = await asyncio.gather(*[_run(s) for s in specs], return_exceptions=True)

        results: list[SpecificationResult] = []
        for spec, outcome in zip(specs, outcomes, strict=True):
            if isinstance(outcome, Exception):
                results.append(SpecificationResult(spec=spec, error=str(outcome)[:120]))
            else:
                results.append(outcome)

        report = MultiverseReport(results=results, direction=1)
        self.replications_run += 1
        return self._apply(hypothesis, report)

    # ------------------------------------------------------------------

    def _apply(self, hypothesis: Hypothesis, report: MultiverseReport) -> dict[str, Any]:
        """Write the multiverse outcome onto the hypothesis and render it."""
        hypothesis.multiverse_fragility = report.fragility
        # Legacy field: reproducibility is now the support rate across forks,
        # which is a meaningful quantity, unlike seed-to-seed variance.
        hypothesis.reproducibility_score = report.support_rate
        hypothesis.replication_results = report.to_dict()["specifications"]

        if not report.robust and report.n_ran:
            hypothesis.limitations.append(
                f"Multiverse fragility {report.fragility:.2f}: the conclusion "
                f"holds in only {report.n_supporting}/{report.n_ran} defensible "
                "analytic specifications."
            )

        narrative = report.render()
        logger.info(
            "Multiverse for '%s': support %.0f%%, fragility %.2f, %d sign flips.",
            (hypothesis.title or "")[:40], 100 * report.support_rate,
            report.fragility, report.sign_flips,
        )

        return {
            "reproducibility_score": report.support_rate,
            "fragility": report.fragility,
            "robust": report.robust,
            "results": [
                f"{r.spec}: effect={r.effect}, p={r.p_value}"
                for r in report.results
            ],
            "consistency": narrative,
            "multiverse": report.to_dict(),
        }

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "reproducibility_score": 0.0,
            "fragility": 0.0,
            "robust": False,
            "results": [],
            "consistency": f"Multiverse analysis not performed: {reason}",
            "multiverse": {},
        }

    # ------------------------------------------------------------------

    async def _get_experiment_code(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
    ) -> str:
        """Recover the analysis code to vary across specifications.

        Prefers code already generated by the ExperimentAgent; otherwise asks
        the LLM for an analysis that calls the injected ``multiverse_analyse``
        harness, so every fork is applied by the same audited code rather than
        re-implemented per specification (which would confound analytic
        choice with code variation).
        """
        for run in reversed(hypothesis.experiment_runs or []):
            code = run.get("code")
            if code:
                return code

        if not self.llm_client:
            return ""

        from utils.llm import get_llm_completion

        prompt = f"""
Research goal: {goal.title}
Hypothesis: {hypothesis.title}
Mechanism: {hypothesis.mechanism}

Write a Python 3 analysis script for a multiverse (specification-curve) study.

A harness is ALREADY injected above your code. Do not redefine it. It provides:

    multiverse_analyse(group_a, group_b)

which applies the current specification (outlier policy, transform,
statistical test, covariate adjustment) and prints the result line itself.

Your script must:
1. Load `data.csv` if present, otherwise construct the two comparison groups
   from documented priors.
2. Build two 1-D numeric arrays: `group_a` (treated) and `group_b` (control).
3. Call `multiverse_analyse(group_a, group_b)` exactly once, as the last
   statement.

Do NOT set a random seed, do NOT choose a statistical test yourself, and do
NOT filter outliers — those are exactly the choices the harness varies.

Output ONLY the Python code in a ```python block.
"""
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False,
                agent_role="code",
            )
            content = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multiverse code generation failed: %s", exc)
            return ""

        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        if "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()


__all__ = ["ReplicationAgent"]
