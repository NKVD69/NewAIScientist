"""
agents/experiment.py — ExperimentAgent for running code-based experiments.

Responsible for:
- Generating experiment code via LLM
- AST-based safety checking before execution
- Running experiments in isolated subprocess
- Fetching real molecular data from PubChem (outside sandbox) and writing
  it as data.csv so the LLM-generated script can analyse real data
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
import sys
from typing import Any

from models.hypothesis import Hypothesis, ResearchGoal
from utils.llm import get_llm_completion
from utils.safety import check_code_safety

from .base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to fetch real molecular / drug data from PubChem (agent-side,
# outside the AST sandbox) and write a local CSV the generated script
# can safely consume via pandas.
# ---------------------------------------------------------------------------

async def _fetch_pubchem_properties(compound_names: list[str], timeout: float = 8.0) -> list[dict]:
    """Query PubChem REST for molecular properties of named compounds.

    Returns a list of dicts with keys: Name, CID, MolecularFormula,
    MolecularWeight, XLogP, HBondDonorCount, HBondAcceptorCount, TPSA.
    """
    import json
    import urllib.parse
    import urllib.request

    results = []
    for name in compound_names[:20]:  # Cap to 20 compounds max
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


def _extract_drug_names_regex(text: str) -> list[str]:
    """Quick regex extraction of drug/compound-like names from hypothesis text.

    Looks for capitalized multi-word patterns commonly seen in drug names,
    and well-known suffixes (-ib, -mab, -cin, -one, -ine, -ide, -ol, -ate).
    """
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
    matches = sorted({m.group(1) for m in drug_suffixes.finditer(text)})
    return matches


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

    # ------------------------------------------------------------------
    # Real-data experiment pipeline
    # ------------------------------------------------------------------

    async def _prepare_real_data(self, hypothesis: Hypothesis) -> list[dict]:
        """Extract drug names from hypothesis and fetch real PubChem data.

        Returns the list of property dicts (may be empty if no drugs found
        or PubChem is unreachable).
        """
        text = " ".join([
            hypothesis.title or "",
            hypothesis.description or "",
            hypothesis.mechanism or "",
            " ".join(hypothesis.testable_predictions or []),
        ])

        # Try LLM-based extraction first, fall back to regex
        drug_names = []
        if self.llm_client:
            try:
                from utils.experiment_sandbox import extract_entities_with_llm
                entities = await extract_entities_with_llm(text, self.llm_client)
                drug_names = entities.get("drugs", [])
            except Exception as exc:
                logger.debug("LLM entity extraction for experiment failed: %s", exc)

        if not drug_names:
            drug_names = _extract_drug_names_regex(text)

        if not drug_names:
            logger.info("No drug/compound names found in hypothesis — skipping real data fetch.")
            return []

        logger.info("Fetching PubChem data for %d compounds: %s", len(drug_names), drug_names[:5])
        return await _fetch_pubchem_properties(drug_names)

    def _write_data_csv(self, data: list[dict], directory: str) -> str | None:
        """Write fetched molecular data as data.csv in the given directory.

        Returns the path to the CSV file, or None if no data.
        """
        if not data:
            return None

        csv_path = os.path.join(directory, "data.csv")
        fieldnames = [
            "Name", "CID", "MolecularFormula", "MolecularWeight",
            "XLogP", "HBondDonorCount", "HBondAcceptorCount", "TPSA",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        logger.info("Wrote %d compounds to %s", len(data), csv_path)
        return csv_path

    async def run_experiment(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        if not self.llm_client:
            return "Simulation skipped: No LLM available for experimental design."

        logger.info("Designing experiment for hypothesis: %s", hypothesis.title)

        # ----- Phase 1: Fetch real data from PubChem -----
        real_data = await self._prepare_real_data(hypothesis)
        has_real_data = len(real_data) > 0

        if has_real_data:
            data_description = (
                f"A file named `data.csv` is available in the working directory.\n"
                f"It contains real molecular properties for {len(real_data)} compounds "
                f"fetched from PubChem.\n"
                f"Columns: Name, CID, MolecularFormula, MolecularWeight, XLogP, "
                f"HBondDonorCount, HBondAcceptorCount, TPSA.\n"
                f"First rows preview: {real_data[:3]}\n\n"
                f"You MUST load this file with `pd.read_csv('data.csv')` and perform "
                f"REAL statistical analysis on this data. Do NOT generate random data."
            )
        else:
            data_description = (
                "No real data file is available. Generate realistic synthetic data "
                "based on known pharmacological properties for the hypothesis."
            )

        # ----- Phase 2: Generate experiment code via LLM -----
        prompt = f"""
        Research Goal: {goal.title}
        Hypothesis: {hypothesis.title}
        Mechanism: {hypothesis.mechanism}
        Predictions: {', '.join(hypothesis.testable_predictions)}

        **Data Context:**
        {data_description}

        You are an AI Scientist tasked with empirically validating this hypothesis.
        Write a Python 3 script that uses standard libraries (numpy, scipy, sklearn, pandas) to run a statistical analysis for the predictions of this hypothesis.
        
        Requirements:
        - If data.csv is available, LOAD it with pandas and perform real analysis.
        - Apply appropriate statistical tests (t-test, correlation, regression, etc.) on the actual molecular properties.
        - It MUST be completely self-contained, do not use UI libraries.
        - It MUST print a clear summary of the results to stdout at the end.
        - Conclude whether the data supports the hypothesis.
        - Include effect sizes, p-values, and confidence intervals where applicable.

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
                # Write real data CSV if available
                if has_real_data:
                    self._write_data_csv(real_data, temp_dir)

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

                    data_source = "REAL PubChem data" if has_real_data else "synthetic data"
                    hypothesis.experimental_results = (
                        f"Experimental Results (using {data_source}):\n{output[:1500]}"
                    )
                    self.experiments_run += 1
                    return hypothesis.experimental_results
                except subprocess.TimeoutExpired:
                    hypothesis.experimental_results = "Experiment simulation timed out after 30 seconds."
                    return hypothesis.experimental_results

        except Exception as e:
            hypothesis.experimental_results = f"Experiment implementation failed: {e}"
            return hypothesis.experimental_results


__all__ = ["ExperimentAgent"]
