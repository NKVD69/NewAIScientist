"""
utils/citation_verifier.py
Citation verifier: extract and validate DOI / arXiv / PMID identifiers
that appear in LLM-generated hypotheses, in order to surface
hallucinated references.

The verifier never fails closed: if a network call errors, the citation
is flagged as ``verified=False`` rather than raising. Downstream agents
(e.g. RankingAgent) can use ``verification_score()`` to penalise
hypotheses whose citations cannot be resolved against the real arXiv,
Crossref, or PubMed catalogues.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

# arXiv: "arXiv:2103.12345", "2103.12345v2", "arxiv.org/abs/1801.0001"
_ARXIV_RE = re.compile(
    r"(?:arxiv[:/]|arxiv\.org/(?:abs|pdf)/)\s*(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
# Bare arXiv IDs adjacent to the word "arxiv" or in citation contexts:
_ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
# DOI: 10.xxxx/...  trailing punctuation stripped at use site.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}\)\]]+)", re.IGNORECASE)
# PMID: "PMID: 12345678" or "pubmed.ncbi.nlm.nih.gov/12345678"
_PMID_RE = re.compile(
    r"(?:pmid[:\s]*|pubmed\.ncbi\.nlm\.nih\.gov/)\s*(\d{5,9})",
    re.IGNORECASE,
)


def _strip_doi_tail(doi: str) -> str:
    """Strip trailing punctuation that often hitchhikes on a DOI."""
    return doi.rstrip(".,;:)]>\"'")


def extract_citation_ids(text: str) -> dict[str, list[str]]:
    """Pull candidate citation identifiers out of free-form text.

    Returns a dict with three deduplicated lists, one per scheme:
    ``arxiv``, ``doi``, ``pmid``. Missing schemes return [].
    """
    if not text:
        return {"arxiv": [], "doi": [], "pmid": []}

    arxiv: set = set()
    for m in _ARXIV_RE.finditer(text):
        arxiv.add(m.group(1))
    # Also catch bare IDs but only when "arxiv" appears nearby
    if "arxiv" in text.lower():
        for m in _ARXIV_BARE_RE.finditer(text):
            arxiv.add(m.group(1))

    doi = {_strip_doi_tail(m.group(1)) for m in _DOI_RE.finditer(text)}
    pmid = {m.group(1) for m in _PMID_RE.finditer(text)}

    return {
        "arxiv": sorted(arxiv),
        "doi": sorted(doi),
        "pmid": sorted(pmid),
    }


# ---------------------------------------------------------------------------
# Citation result
# ---------------------------------------------------------------------------

@dataclass
class CitationResult:
    """Outcome of verifying a single citation identifier."""
    identifier: str
    type: str            # "arxiv" | "doi" | "pmid"
    verified: bool
    source_url: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Network helper
# ---------------------------------------------------------------------------

def _head_or_get(url: str, timeout: float) -> int:
    """Synchronous HEAD then GET fallback. Returns HTTP status, or 0 on error."""
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(
                url,
                method=method,
                headers={"User-Agent": "NewAIScientist-CitationVerifier/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return int(resp.status)
        except urllib.error.HTTPError as e:
            # Some servers reject HEAD with 405 — retry with GET.
            if method == "HEAD" and e.code in (403, 405, 501):
                continue
            return int(e.code)
        except Exception:
            if method == "HEAD":
                continue
            return 0
    return 0


async def _resolve(url: str, timeout: float = 5.0) -> int:
    """Async wrapper around _head_or_get; returns the status code."""
    try:
        return await asyncio.to_thread(_head_or_get, url, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Citation verify network error for %s: %s", url, exc)
        return 0


def _is_ok(status: int) -> bool:
    return 200 <= status < 400


# ---------------------------------------------------------------------------
# Per-scheme verifiers
# ---------------------------------------------------------------------------

async def verify_arxiv_id(arxiv_id: str, timeout: float = 5.0) -> CitationResult:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    status = await _resolve(url, timeout)
    return CitationResult(
        identifier=arxiv_id,
        type="arxiv",
        verified=_is_ok(status),
        source_url=url,
        error="" if _is_ok(status) else f"HTTP {status}",
    )


async def verify_doi(doi: str, timeout: float = 5.0) -> CitationResult:
    url = f"https://doi.org/{doi}"
    status = await _resolve(url, timeout)
    return CitationResult(
        identifier=doi,
        type="doi",
        verified=_is_ok(status),
        source_url=url,
        error="" if _is_ok(status) else f"HTTP {status}",
    )


async def verify_pmid(pmid: str, timeout: float = 5.0) -> CitationResult:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    status = await _resolve(url, timeout)
    return CitationResult(
        identifier=pmid,
        type="pmid",
        verified=_is_ok(status),
        source_url=url,
        error="" if _is_ok(status) else f"HTTP {status}",
    )


# ---------------------------------------------------------------------------
# Aggregate APIs
# ---------------------------------------------------------------------------

async def verify_text(
    text: str,
    timeout: float = 5.0,
    max_concurrent: int = 8,
) -> list[CitationResult]:
    """Extract every citation identifier in *text* and verify them in parallel."""
    ids = extract_citation_ids(text)

    sem = asyncio.Semaphore(max_concurrent)

    async def _bound(coro):
        async with sem:
            return await coro

    coros = []
    coros += [_bound(verify_arxiv_id(a, timeout)) for a in ids["arxiv"]]
    coros += [_bound(verify_doi(d, timeout)) for d in ids["doi"]]
    coros += [_bound(verify_pmid(p, timeout)) for p in ids["pmid"]]

    if not coros:
        return []

    return await asyncio.gather(*coros)


def hypothesis_text(hyp: Any) -> str:
    """Concatenate all hypothesis fields where citations may appear."""
    parts = [
        getattr(hyp, "title", "") or "",
        getattr(hyp, "description", "") or "",
        getattr(hyp, "mechanism", "") or "",
        getattr(hyp, "reasoning", "") or "",
        " ".join(getattr(hyp, "cited_papers", []) or []),
        " ".join(getattr(hyp, "grounding_evidence", []) or []),
    ]
    return "\n".join(p for p in parts if p)


async def verify_hypothesis(hyp: Any, timeout: float = 5.0) -> list[CitationResult]:
    """Verify every citation found inside a Hypothesis object."""
    return await verify_text(hypothesis_text(hyp), timeout=timeout)


def verification_score(results: list[CitationResult]) -> float:
    """Fraction of resolved citations.

    Returns 1.0 when no citations were extracted (treated as a neutral case
    rather than a hallucination flag).
    """
    if not results:
        return 1.0
    verified = sum(1 for r in results if r.verified)
    return verified / len(results)


def apply_verification_penalty(
    elo_rating: float,
    results: list[CitationResult],
    max_penalty: float = 200.0,
) -> float:
    """Subtract from the Elo a penalty proportional to unverified citations.

    A hypothesis with all citations resolved (or no citations) keeps its Elo;
    one with every citation hallucinated loses ``max_penalty`` points.
    """
    score = verification_score(results)
    return elo_rating - max_penalty * (1.0 - score)


__all__ = [
    "CitationResult",
    "extract_citation_ids",
    "verify_arxiv_id",
    "verify_doi",
    "verify_pmid",
    "verify_text",
    "verify_hypothesis",
    "verification_score",
    "apply_verification_penalty",
    "hypothesis_text",
]
