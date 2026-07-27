"""
utils/experiment_sandbox.py
Lightweight feasibility checks for proposed experiments.

Two independent checks the ExperimentAgent can call before / instead of
running an actual simulation:

1. ``estimate_required_n``: closed-form Cohen-style power analysis for a
   two-sample t-test. Returns the per-group sample size required to
   detect ``effect_size`` at a given alpha and power. Pure math; no scipy.

2. ``validate_entities``: HTTP-mediated existence check for biomedical
   identifiers (UniProt accessions, PubChem CIDs, gene symbols via NCBI
   Entrez when available). Each entity is asynchronously resolved and
   returned with a verified flag — same shape as the citation verifier.

Both functions degrade gracefully: a missing optional dependency or a
network error returns a permissive default rather than raising.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Power analysis — two-sample t-test
# ---------------------------------------------------------------------------

# Two-sided z-quantile lookups. Pre-computed to avoid importing scipy for
# such a tiny operation; covers the alphas / powers used in practice.
_Z_TWO_SIDED = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758, 0.001: 3.2905}
_Z_ONE_SIDED = {0.50: 0.0000, 0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def estimate_required_n(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Per-group sample size for a two-sample two-sided t-test.

    Uses the standard normal approximation (no scipy required):

        n = 2 * ((z_{α/2} + z_{1-β}) / d)^2

    Parameters
    ----------
    effect_size
        Standardised mean difference (Cohen's d). Negative values are
        treated as their absolute value.
    alpha
        Type-I error rate. Defaults to 0.05.
    power
        Desired statistical power (1 − β). Defaults to 0.80.

    Returns
    -------
    Smallest integer n such that the two-sample test attains ``power``.
    Returns ``-1`` for an effectively-zero effect (undetectable).
    """
    d = abs(float(effect_size))
    if d < 1e-9:
        return -1
    z_alpha = _Z_TWO_SIDED.get(alpha)
    if z_alpha is None:
        # Fall back to the closest bracketed value.
        z_alpha = _Z_TWO_SIDED[min(_Z_TWO_SIDED, key=lambda a: abs(a - alpha))]
    z_power = _Z_ONE_SIDED.get(power)
    if z_power is None:
        z_power = _Z_ONE_SIDED[min(_Z_ONE_SIDED, key=lambda p: abs(p - power))]
    n = 2.0 * ((z_alpha + z_power) / d) ** 2
    return int(math.ceil(n))


# ---------------------------------------------------------------------------
# Entity validation
# ---------------------------------------------------------------------------

# UniProt accession (canonical pattern, e.g. P04637, Q8WZ42)
_UNIPROT_RE = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")
# PubChem CID — "CID 12345" or "CID:12345"
_PUBCHEM_RE = re.compile(r"\bCID[:\s]+(\d{1,9})\b", re.IGNORECASE)
# Gene symbols — common human gene patterns (uppercase, 2-10 chars, optional digits)
_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
# Known false-positive gene symbols to filter out
_GENE_FALSE_POSITIVES = frozenset({
    "DNA", "RNA", "ATP", "GTP", "ADP", "GDP", "AMP", "GMP",
    "FDA", "AML", "ALL", "CLL", "CML", "MDS", "NHL",
    "PCR", "ELISA", "FACS", "HPLC", "NMR", "CRISPR",
    "IC50", "EC50", "ED50", "LD50",
    "RCT", "ROS", "NAD", "NADH", "FADH", "COX", "SOD",
    "BMI", "ECG", "MRI", "PET", "CAT", "EEG",
    "WHO", "NIH", "CDC", "EMA", "NICE",
    "ANOVA", "ANCOVA", "MANOVA",
    "JSON", "API", "URL", "CSV", "PDF", "LLM",
    "THE", "FOR", "AND", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD",
    "CAN", "MAY", "WILL", "SHALL", "MUST", "VIA", "PER",
    "NEW", "OLD", "LOW", "HIGH", "USE", "SET", "GET",
})


@dataclass
class EntityResult:
    identifier: str
    type: str           # "uniprot" | "pubchem" | "gene" | "drug"
    verified: bool
    source_url: str = ""
    error: str = ""
    details: str = ""   # Extra info from verification (e.g. full gene name)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Pull UniProt / PubChem / gene-symbol candidates from free text.

    The result is keyed by scheme; values are deduplicated and sorted.
    """
    if not text:
        return {"uniprot": [], "pubchem": [], "genes": [], "drugs": []}
    uniprot = sorted({m.group(1) for m in _UNIPROT_RE.finditer(text)})
    pubchem = sorted({m.group(1) for m in _PUBCHEM_RE.finditer(text)})
    return {"uniprot": uniprot, "pubchem": pubchem, "genes": [], "drugs": []}


async def extract_entities_with_llm(
    text: str,
    llm_client=None,
) -> dict[str, list[str]]:
    """Extract biomedical entities using LLM for better coverage.

    Falls back to regex-only extraction when no LLM client is available.
    The LLM identifies gene symbols, drug names, protein names, and
    molecular identifiers that regex patterns may miss.
    """
    # Start with regex-based extraction
    base = extract_entities(text)

    if llm_client is None:
        return base

    prompt = f"""Extract all biomedical entities from the following scientific text.
Return a JSON object with these keys:
- "genes": list of human gene symbols (e.g. TP53, BRCA1, FLT3)
- "drugs": list of drug or compound names (e.g. aspirin, imatinib, venetoclax)
- "uniprot": list of UniProt accession IDs (e.g. P04637)
- "pubchem_cids": list of PubChem CID numbers as strings

Only include entities you are confident about. Do NOT hallucinate.

Text: \"{text[:2000]}\"

Return ONLY the JSON object."""

    try:
        from utils.llm import get_llm_completion, parse_json_response
        response = await get_llm_completion(
            llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            json_mode=True,
        )
        data = parse_json_response(response.choices[0].message.content)

        # Merge LLM results with regex results (dedup)
        llm_genes = [g.strip().upper() for g in data.get("genes", []) if g.strip()]
        llm_drugs = [d.strip() for d in data.get("drugs", []) if d.strip()]
        llm_uniprot = [u.strip() for u in data.get("uniprot", []) if u.strip()]
        llm_pubchem = [str(c).strip() for c in data.get("pubchem_cids", []) if str(c).strip()]

        base["genes"] = sorted(set(llm_genes))
        base["drugs"] = sorted(set(llm_drugs))
        base["uniprot"] = sorted(set(base["uniprot"]) | set(llm_uniprot))
        base["pubchem"] = sorted(set(base["pubchem"]) | set(llm_pubchem))

    except Exception as exc:
        logger.warning("LLM entity extraction failed: %s — using regex only.", exc)

    return base


def _http_status(url: str, timeout: float) -> int:
    try:
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": "NewAIScientist-EntityValidator/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:
        return 0


def _http_get_json(url: str, timeout: float) -> dict | None:
    """Fetch a URL and parse the response as JSON. Returns None on failure."""
    try:
        req = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent": "NewAIScientist-EntityValidator/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


async def _resolve(url: str, timeout: float) -> int:
    try:
        return await asyncio.to_thread(_http_status, url, timeout)
    except Exception:
        return 0


async def _resolve_json(url: str, timeout: float) -> dict | None:
    try:
        return await asyncio.to_thread(_http_get_json, url, timeout)
    except Exception:
        return None


async def verify_uniprot(accession: str, timeout: float = 5.0) -> EntityResult:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    status = await _resolve(url, timeout)
    return EntityResult(
        identifier=accession,
        type="uniprot",
        verified=200 <= status < 300,
        source_url=url,
        error="" if 200 <= status < 300 else f"HTTP {status}",
    )


async def verify_pubchem(cid: str, timeout: float = 5.0) -> EntityResult:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/cid/{cid}/property/MolecularFormula/JSON"
    status = await _resolve(url, timeout)
    return EntityResult(
        identifier=cid,
        type="pubchem",
        verified=200 <= status < 300,
        source_url=url,
        error="" if 200 <= status < 300 else f"HTTP {status}",
    )


async def verify_gene(gene_symbol: str, timeout: float = 5.0) -> EntityResult:
    """Verify a human gene symbol exists via NCBI Entrez esearch.

    Queries the NCBI Gene database for the symbol restricted to Homo sapiens.
    Returns an EntityResult with verified=True if at least one match is found.
    """
    safe_symbol = urllib.parse.quote(gene_symbol)
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=gene&term={safe_symbol}%5BGene+Name%5D+AND+Homo+sapiens%5BOrganism%5D"
        f"&retmode=json&retmax=1"
    )
    data = await _resolve_json(url, timeout)
    if data is None:
        return EntityResult(
            identifier=gene_symbol, type="gene", verified=False,
            source_url=url, error="Network error or timeout",
        )

    result = data.get("esearchresult", {})
    count = int(result.get("count", 0))
    id_list = result.get("idlist", [])

    return EntityResult(
        identifier=gene_symbol,
        type="gene",
        verified=count > 0,
        source_url=url,
        error="" if count > 0 else "Gene not found in NCBI for Homo sapiens",
        details=f"NCBI Gene ID: {id_list[0]}" if id_list else "",
    )


async def verify_drug_name(drug_name: str, timeout: float = 5.0) -> EntityResult:
    """Verify a drug/compound name exists via PubChem name search.

    Queries the PubChem REST API by compound name to check if the molecule
    exists. Returns an EntityResult with verified=True if found.
    """
    safe_name = urllib.parse.quote(drug_name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{safe_name}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
    )
    data = await _resolve_json(url, timeout)
    if data is None:
        return EntityResult(
            identifier=drug_name, type="drug", verified=False,
            source_url=url, error="Network error or timeout",
        )

    props = data.get("PropertyTable", {}).get("Properties", [])
    if props:
        p = props[0]
        details = (
            f"CID: {p.get('CID', '?')}, "
            f"Formula: {p.get('MolecularFormula', '?')}, "
            f"MW: {p.get('MolecularWeight', '?')}"
        )
        return EntityResult(
            identifier=drug_name, type="drug", verified=True,
            source_url=url, details=details,
        )
    else:
        return EntityResult(
            identifier=drug_name, type="drug", verified=False,
            source_url=url, error="Drug/compound not found in PubChem",
        )


async def validate_entities(
    text: str,
    timeout: float = 5.0,
    max_concurrent: int = 4,
    llm_client=None,
) -> list[EntityResult]:
    """Verify every biomedical entity in *text* concurrently.

    When ``llm_client`` is provided, uses LLM-powered extraction for
    better coverage of gene symbols and drug names (beyond regex).
    Otherwise falls back to regex-only extraction of UniProt/PubChem IDs.
    """
    if llm_client is not None:
        ids = await extract_entities_with_llm(text, llm_client)
    else:
        ids = extract_entities(text)

    sem = asyncio.Semaphore(max_concurrent)

    async def _bound(coro):
        async with sem:
            return await coro

    coros = []
    coros += [_bound(verify_uniprot(a, timeout)) for a in ids.get("uniprot", [])]
    coros += [_bound(verify_pubchem(c, timeout)) for c in ids.get("pubchem", [])]
    coros += [_bound(verify_gene(g, timeout)) for g in ids.get("genes", [])]
    coros += [_bound(verify_drug_name(d, timeout)) for d in ids.get("drugs", [])]
    if not coros:
        return []
    return await asyncio.gather(*coros)


def feasibility_summary(
    required_n: int,
    entity_results: list[EntityResult],
) -> dict[str, object]:
    """Compact dict the ExperimentAgent can include in its report."""
    total = len(entity_results)
    verified = sum(1 for r in entity_results if r.verified)
    unverified = [r for r in entity_results if not r.verified]
    return {
        "required_n_per_group": required_n,
        "n_entities_total": total,
        "n_entities_verified": verified,
        "entity_verification_rate": (verified / total) if total else 1.0,
        "entities": [
            {
                "id": r.identifier,
                "type": r.type,
                "verified": r.verified,
                "source": r.source_url,
                "details": r.details,
            }
            for r in entity_results
        ],
        "unverified_entities": [
            {"id": r.identifier, "type": r.type, "error": r.error}
            for r in unverified
        ],
    }


def format_entity_report(entity_results: list[EntityResult]) -> str:
    """Format entity validation results as a human-readable report.

    Used by the ReflectionAgent to inject entity validation context
    into the multi-agent review committee prompts.
    """
    if not entity_results:
        return "No biomedical entities were detected for verification."

    lines = ["## Entity Validation Report"]

    verified = [r for r in entity_results if r.verified]
    unverified = [r for r in entity_results if not r.verified]

    if verified:
        lines.append(f"\n✅ **Verified ({len(verified)}):**")
        for r in verified:
            detail = f" — {r.details}" if r.details else ""
            lines.append(f"  - [{r.type.upper()}] {r.identifier}{detail}")

    if unverified:
        lines.append(f"\n❌ **Unverified / Not Found ({len(unverified)}):**")
        for r in unverified:
            lines.append(f"  - [{r.type.upper()}] {r.identifier}: {r.error}")

    rate = (len(verified) / len(entity_results)) * 100 if entity_results else 0
    lines.append(f"\n**Verification Rate:** {rate:.0f}% ({len(verified)}/{len(entity_results)})")

    return "\n".join(lines)


__all__ = [
    "EntityResult",
    "estimate_required_n",
    "extract_entities",
    "extract_entities_with_llm",
    "feasibility_summary",
    "format_entity_report",
    "validate_entities",
    "verify_drug_name",
    "verify_gene",
    "verify_pubchem",
    "verify_uniprot",
]
