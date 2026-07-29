"""
agents/experiment.py — ExperimentAgent: typed, sandboxed, adjudicated experiments.

Two changes of nature relative to the previous implementation.

**1. Experiments are typed, and simulations cannot corroborate.**

The old agent had a default path that told the LLM to "generate realistic
synthetic data", ran t-tests on it, and asked the model to conclude whether
the data supported the hypothesis. Those p-values measure the internal
consistency of the generator with itself; they carry zero external
information, yet they arrived downstream dressed in the full apparatus of a
result (effect sizes, confidence intervals) and were rendered as Results by
the WritingAgent.

Every run now carries an ``ExperimentKind``. A ``DRY_RUN_SIMULATION`` may
*refute* a hypothesis (internal incoherence is informative) but can never
*corroborate* one -- enforced in ``utils.adjudication``. When no relevant
data source can be reached, the run is ``INFEASIBLE`` and says so, instead
of manufacturing agreement.

**2. Execution is isolated by the kernel, not by an import blocklist.**

``check_code_safety`` inspected only ``Import``/``ImportFrom`` nodes; ten
bypasses were verified against the shipped code, all passing (``importlib``,
``__import__``, ``http.client`` for egress, ``open()`` for writes, ``exec``,
``__subclasses__`` traversal, infinite loops, memory bombs). It is retained
here as an early *quality* filter and explicitly demoted: the security
boundary is ``utils.sandbox_runner``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

from models.experiment import ExperimentKind, ExperimentRun
from models.hypothesis import Hypothesis, ResearchGoal
from utils.adjudication import (
    RESULTS_CONTRACT,
    adjudicate,
    build_prediction_contract,
    format_verdict_report,
    parse_measurements,
)
from utils.llm import get_llm_completion
from utils.safety import check_code_safety
from utils.sandbox_runner import SandboxPolicy, isolation_report, run_sandboxed

from .base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data acquisition (agent-side, outside the sandbox)
# ---------------------------------------------------------------------------

async def _fetch_pubchem_properties(compound_names: list[str], timeout: float = 8.0) -> list[dict]:
    """Query PubChem REST for molecular properties of named compounds.

    NOTE ON RELEVANCE: these are physicochemical descriptors (MW, XLogP,
    TPSA...). They are real data, but for a proliferation-inhibition or
    target-engagement hypothesis they are close to irrelevant -- correlating
    TPSA with leukaemic cell growth tests Lipinski's rules, not the
    hypothesis. ``_classify_experiment`` therefore refuses to promote a
    PubChem-only run to REAL_DATA_ANALYSIS unless the registered predictions
    actually concern physicochemical quantities. Richer sources (ChEMBL
    bioactivities, DepMap dependencies) belong here; see ``_DATA_SOURCES``.
    """
    import asyncio
    import json
    import urllib.parse
    import urllib.request

    results = []
    for name in compound_names[:20]:
        safe = urllib.parse.quote(name.strip())
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe}"
            f"/property/MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,"
            f"HBondAcceptorCount,TPSA/JSON"
        )
        try:
            def _get(u=url):
                req = urllib.request.Request(
                    u, method="GET",
                    headers={"User-Agent": "NewAIScientist-Experiment/1.0"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            data = await asyncio.to_thread(_get)
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if props_list:
                p = props_list[0]
                results.append({
                    "Name": name.strip(),
                    "CID": p.get("CID", ""),
                    "MolecularFormula": p.get("MolecularFormula", ""),
                    "MolecularWeight": p.get("MolecularWeight", ""),
                    "XLogP": p.get("XLogP", ""),
                    "HBondDonorCount": p.get("HBondDonorCount", ""),
                    "HBondAcceptorCount": p.get("HBondAcceptorCount", ""),
                    "TPSA": p.get("TPSA", ""),
                })
        except Exception as exc:
            logger.debug("PubChem lookup failed for '%s': %s", name, exc)
    return results


async def _fetch_chembl_activities(compound_names: list[str], timeout: float = 10.0) -> list[dict]:
    """Fetch measured bioactivities (IC50/Ki/EC50) from ChEMBL.

    Unlike PubChem descriptors, these are *assay outcomes* -- the kind of
    quantity a pharmacological hypothesis actually predicts, and therefore
    the kind that can genuinely corroborate or refute one.
    """
    import asyncio
    import json
    import urllib.parse
    import urllib.request

    rows: list[dict] = []
    for name in compound_names[:10]:
        try:
            lookup = (
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
                f"?q={urllib.parse.quote(name.strip())}&limit=1"
            )

            def _get(u):
                req = urllib.request.Request(
                    u, headers={"User-Agent": "NewAIScientist-Experiment/1.0",
                                "Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            found = await asyncio.to_thread(_get, lookup)
            molecules = found.get("molecules", [])
            if not molecules:
                continue
            chembl_id = molecules[0].get("molecule_chembl_id")
            if not chembl_id:
                continue

            act_url = (
                "https://www.ebi.ac.uk/chembl/api/data/activity.json"
                f"?molecule_chembl_id={chembl_id}"
                "&standard_type__in=IC50,Ki,EC50&limit=25"
            )
            acts = await asyncio.to_thread(_get, act_url)
            for a in acts.get("activities", []):
                if a.get("standard_value") in (None, ""):
                    continue
                rows.append({
                    "Name": name.strip(),
                    "ChEMBLID": chembl_id,
                    "Target": a.get("target_pref_name", ""),
                    "TargetChEMBLID": a.get("target_chembl_id", ""),
                    "StandardType": a.get("standard_type", ""),
                    "StandardValue": a.get("standard_value", ""),
                    "StandardUnits": a.get("standard_units", ""),
                    "AssayDescription": (a.get("assay_description") or "")[:200],
                })
        except Exception as exc:
            logger.debug("ChEMBL lookup failed for '%s': %s", name, exc)
    return rows


#: Registry of real-data acquisition backends, tried in order of relevance.
#: Each entry is (name, fetcher, csv_filename, "measures assay outcomes?").
_DATA_SOURCES = [
    ("ChEMBL", _fetch_chembl_activities, "activities.csv", True),
    ("PubChem", _fetch_pubchem_properties, "properties.csv", False),
]

#: Quantities for which PubChem descriptors are a legitimate test.
_PHYSCHEM_QUANTITIES = re.compile(
    r"molecular\s*weight|logp|xlogp|tpsa|polar\s*surface|h.?bond|lipinski|"
    r"solubilit|permeabilit",
    re.IGNORECASE,
)


def _extract_drug_names_regex(text: str) -> list[str]:
    """Regex extraction of drug/compound-like names from hypothesis text."""
    drug_suffixes = re.compile(
        r"\b([A-Z][a-z]{2,}(?:inib|tinib|zomib|cisib|fenib|"
        r"mab|ximab|zumab|mumab|"
        r"mycin|rubicin|vastatin|prazole|"
        r"done|olone|sone|"
        r"pine|tidine|azine|"
        r"mide|azide|oxide|"
        r"olol|alol|"
        r"pril|artan|"
        r"oxacin|cycline|"
        r"clax|metinib|"
        r"etine|oxetine))\b",
        re.IGNORECASE,
    )
    return sorted({m.group(1) for m in drug_suffixes.finditer(text)})


def _rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ExperimentAgent(BaseAgent):
    """Generates, sandboxes and adjudicates code-based experiments."""

    name = "Experiment"

    def __init__(self, use_local_llm: bool = True,
                 allow_simulation: bool = True,
                 policy: SandboxPolicy | None = None):
        super().__init__(use_local_llm=use_local_llm)
        self.experiments_run = 0
        #: When False, a hypothesis with no real data source is marked
        #: INFEASIBLE rather than simulated. Recommended for production runs.
        self.allow_simulation = allow_simulation
        self.policy = policy or SandboxPolicy.from_env()
        self.runs: list[ExperimentRun] = []

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    async def feasibility_check(
        self,
        hypothesis: Hypothesis,
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict[str, Any]:
        """Estimate sample size and validate biomedical entities mentioned."""
        from utils.experiment_sandbox import (
            estimate_required_n,
            feasibility_summary,
            validate_entities,
        )

        text = self._hypothesis_text(hypothesis)
        try:
            entities = await validate_entities(text, llm_client=self.llm_client)
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

    @staticmethod
    def _hypothesis_text(hypothesis: Hypothesis) -> str:
        return " ".join([
            hypothesis.title or "",
            hypothesis.description or "",
            hypothesis.mechanism or "",
            " ".join(hypothesis.testable_predictions or []),
        ])

    # ------------------------------------------------------------------
    # Data acquisition & experiment classification
    # ------------------------------------------------------------------

    async def _acquire_data(self, hypothesis: Hypothesis) -> tuple[str, str, bool]:
        """Try each real-data source in order of relevance.

        Returns ``(source_name, csv_text, measures_outcomes)``. Empty source
        name means no external data could be obtained.
        """
        text = self._hypothesis_text(hypothesis)

        drug_names: list[str] = []
        if self.llm_client:
            try:
                from utils.experiment_sandbox import extract_entities_with_llm
                entities = await extract_entities_with_llm(text, self.llm_client)
                drug_names = entities.get("drugs", [])
            except Exception as exc:
                logger.debug("LLM entity extraction failed: %s", exc)
        if not drug_names:
            drug_names = _extract_drug_names_regex(text)

        if not drug_names:
            logger.info("No compounds identified in hypothesis — no data source applicable.")
            return "", "", False

        for source_name, fetcher, _filename, measures_outcomes in _DATA_SOURCES:
            try:
                rows = await fetcher(drug_names)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s fetch failed: %s", source_name, exc)
                continue
            if rows:
                logger.info("Acquired %d rows from %s.", len(rows), source_name)
                return source_name, _rows_to_csv(rows), measures_outcomes

        return "", "", False

    def _classify_experiment(
        self,
        hypothesis: Hypothesis,
        source_name: str,
        measures_outcomes: bool,
    ) -> tuple[ExperimentKind, str]:
        """Decide what kind of evidence this run is capable of producing.

        The guard that matters: descriptor data (PubChem) is only allowed to
        count as REAL_DATA_ANALYSIS when the registered predictions actually
        concern physicochemical quantities. Otherwise we have real numbers
        that are irrelevant to the claim, which is epistemically no better
        than synthetic data and should not be allowed to corroborate.
        """
        if not source_name:
            if self.allow_simulation:
                return (
                    ExperimentKind.DRY_RUN_SIMULATION,
                    "no external data source reachable; simulation may refute "
                    "(internal incoherence) but cannot corroborate",
                )
            return (
                ExperimentKind.INFEASIBLE,
                "no external data source reachable and simulation is disabled",
            )

        if measures_outcomes:
            return (
                ExperimentKind.REAL_DATA_ANALYSIS,
                f"{source_name} supplies measured assay outcomes",
            )

        registered = " ".join(
            f"{p.quantity} {p.unit}" for p in (hypothesis.falsifiable_predictions or [])
        )
        if registered and _PHYSCHEM_QUANTITIES.search(registered):
            return (
                ExperimentKind.REAL_DATA_ANALYSIS,
                f"{source_name} descriptors match the registered physicochemical quantities",
            )

        if self.allow_simulation:
            return (
                ExperimentKind.DRY_RUN_SIMULATION,
                f"{source_name} returned only physicochemical descriptors, which do not "
                f"bear on the registered quantities ({registered[:80] or 'none'}); "
                "downgraded to simulation — cannot corroborate",
            )
        return (
            ExperimentKind.INFEASIBLE,
            f"{source_name} data is irrelevant to the registered quantities "
            "and simulation is disabled",
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
        kind: ExperimentKind,
        source_name: str,
        csv_preview: str,
    ) -> str:
        if source_name:
            data_block = (
                f"A file named `data.csv` sits in your working directory, containing "
                f"real records fetched from {source_name}.\n"
                f"First lines:\n{csv_preview}\n\n"
                "You MUST load it with `pd.read_csv('data.csv')` and analyse THAT data. "
                "Do not generate random data."
            )
        else:
            data_block = (
                "No external data is available. You may simulate data from documented "
                "pharmacological priors, but be aware this run is classified as a "
                "SIMULATION: it can only demonstrate internal incoherence, it cannot "
                "provide evidence FOR the hypothesis. State your simulation "
                "assumptions explicitly in the output."
            )

        return f"""
Research Goal: {goal.title}
Hypothesis: {hypothesis.title}
Mechanism: {hypothesis.mechanism}

**Experiment classification:** {kind.value}
{"This run CAN corroborate or refute." if kind.can_corroborate
 else "This run can only REFUTE. Agreement with the prediction earns no credit."}

**Data context:**
{data_block}

**{build_prediction_contract(hypothesis)}**

Write a self-contained Python 3 script (numpy, scipy, pandas, sklearn) that
measures the pre-registered quantities above.

Requirements:
- No UI libraries, no network access (the sandbox has none), no file writes
  outside the working directory.
- Apply appropriate statistical tests; report effect sizes and CIs.
- Print a readable human summary of what you did and found.
- Report measurements honestly, including ones that contradict the
  hypothesis. A refutation is a valid scientific result.

{RESULTS_CONTRACT}

Output ONLY the Python code inside a ```python block.
"""

    @staticmethod
    def _extract_code(content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        if "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_experiment(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        """Design, sandbox, execute and adjudicate one experiment.

        Returns a human-readable summary for backwards compatibility. The
        authoritative artefact is the ``ExperimentRun`` appended to
        ``hypothesis.experiment_runs`` — all decision logic must read that,
        never the string.
        """
        run = ExperimentRun(hypothesis_id=hypothesis.id)

        if not self.llm_client:
            run.kind = ExperimentKind.INFEASIBLE
            run.error = "no LLM available to design the experiment"
            return self._finalise(hypothesis, run)

        # --- 1. Acquire data and classify what this run can conclude -----
        source_name, csv_text, measures_outcomes = await self._acquire_data(hypothesis)
        kind, rationale = self._classify_experiment(hypothesis, source_name, measures_outcomes)
        run.kind = kind
        run.data_source = source_name or "synthetic"

        logger.info("Experiment for '%s' classified as %s — %s",
                    (hypothesis.title or "")[:40], kind.value, rationale)

        if kind is ExperimentKind.INFEASIBLE:
            run.error = rationale
            return self._finalise(hypothesis, run)

        # --- 2. Generate the script --------------------------------------
        preview = "\n".join(csv_text.splitlines()[:4]) if csv_text else ""
        prompt = self._build_prompt(hypothesis, goal, kind, source_name, preview)

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False,
                agent_role="code",
            )
            code = self._extract_code(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            run.error = f"code generation failed: {exc}"
            return self._finalise(hypothesis, run)

        if not code:
            run.error = "LLM returned no executable code"
            return self._finalise(hypothesis, run)

        # --- 3. Quality pre-filter (NOT a security boundary) -------------
        # Kept because it catches obviously off-task code cheaply. The actual
        # isolation is the sandbox below; see utils/sandbox_runner.py for why
        # an AST import blocklist cannot be relied upon.
        is_clean, reason = check_code_safety(code)
        if not is_clean:
            logger.info("Quality filter rejected generated code: %s", reason)
            run.error = f"generated code rejected by quality filter: {reason}"
            return self._finalise(hypothesis, run)

        # --- 4. Execute under kernel-enforced isolation -------------------
        input_files = {"data.csv": csv_text} if csv_text else None
        result = await run_sandboxed(code, input_files=input_files, policy=self.policy)

        run.sandbox_backend = result.backend
        run.code_sha256 = result.code_sha256
        run.code = code
        run.stdout = result.stdout[:8000]
        run.stderr = result.stderr[:4000]
        run.exit_code = result.exit_code
        run.duration_s = result.duration_s

        if result.blocked:
            run.error = f"execution refused: {result.error}"
            logger.error("Sandbox refused execution: %s", result.error)
            return self._finalise(hypothesis, run)
        if result.timed_out:
            run.error = result.error or "execution timed out"
            return self._finalise(hypothesis, run)

        # --- 5. Parse measurements and adjudicate against pre-registration -
        run.measurements = parse_measurements(result.stdout)
        if not run.measurements:
            logger.warning(
                "Script produced no parseable RESULTS_JSON line; all predictions "
                "will be recorded as UNTESTED.",
            )
        run.verdicts = adjudicate(hypothesis, run.measurements, kind=kind)

        self.experiments_run += 1
        return self._finalise(hypothesis, run)

    # ------------------------------------------------------------------

    def _finalise(self, hypothesis: Hypothesis, run: ExperimentRun) -> str:
        """Attach the run to the hypothesis and render the legacy string."""
        self.runs.append(run)
        hypothesis.experiment_runs.append(run.to_dict())
        hypothesis.verdicts = [v.to_dict() for v in run.verdicts]

        # Accumulate signed evidence across runs rather than overwriting, so a
        # later inconclusive run cannot erase an earlier refutation.
        prior = hypothesis.empirical_support
        weight = run.evidential_weight
        hypothesis.empirical_support = max(-1.0, min(1.0, prior + weight))

        report = format_verdict_report(run.verdicts)
        parts = [
            f"Experiment [{run.kind.value}] via {run.data_source} "
            f"(sandbox: {run.sandbox_backend or 'n/a'})",
            run.summary(),
        ]
        if run.error:
            parts.append(f"Error: {run.error}")
        if run.stdout:
            parts.append("--- stdout ---\n" + run.stdout[:1500])
        parts.append(report)

        hypothesis.experimental_results = "\n\n".join(parts)
        return hypothesis.experimental_results

    # ------------------------------------------------------------------

    def isolation_status(self) -> dict:
        """Expose the isolation actually in force, for the UI and logs."""
        return isolation_report()


__all__ = ["ExperimentAgent"]
