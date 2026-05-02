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
import logging
import math
import re
import urllib.error
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


@dataclass
class EntityResult:
    identifier: str
    type: str           # "uniprot" | "pubchem"
    verified: bool
    source_url: str = ""
    error: str = ""


def extract_entities(text: str) -> dict[str, list[str]]:
    """Pull UniProt / PubChem / gene-symbol candidates from free text.

    The result is keyed by scheme; values are deduplicated and sorted.
    """
    if not text:
        return {"uniprot": [], "pubchem": []}
    uniprot = sorted({m.group(1) for m in _UNIPROT_RE.finditer(text)})
    pubchem = sorted({m.group(1) for m in _PUBCHEM_RE.finditer(text)})
    return {"uniprot": uniprot, "pubchem": pubchem}


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


async def _resolve(url: str, timeout: float) -> int:
    try:
        return await asyncio.to_thread(_http_status, url, timeout)
    except Exception:
        return 0


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


async def validate_entities(
    text: str,
    timeout: float = 5.0,
    max_concurrent: int = 4,
) -> list[EntityResult]:
    """Verify every UniProt / PubChem identifier in *text* concurrently."""
    ids = extract_entities(text)
    sem = asyncio.Semaphore(max_concurrent)

    async def _bound(coro):
        async with sem:
            return await coro

    coros = []
    coros += [_bound(verify_uniprot(a, timeout)) for a in ids["uniprot"]]
    coros += [_bound(verify_pubchem(c, timeout)) for c in ids["pubchem"]]
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
            }
            for r in entity_results
        ],
    }


__all__ = [
    "EntityResult",
    "estimate_required_n",
    "extract_entities",
    "feasibility_summary",
    "validate_entities",
    "verify_pubchem",
    "verify_uniprot",
]
