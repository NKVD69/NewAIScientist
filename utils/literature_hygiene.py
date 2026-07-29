"""
utils/literature_hygiene.py — DOI deduplication, retraction screening, recency weighting.

Three gaps in the retrieval layer, in rough order of severity.

**Retractions.** A system that generates biomedical hypotheses had no
retraction check at all. Grounding a hypothesis on a retracted paper is a
serious and entirely avoidable failure mode — the data exists, in PubMed's
``PublicationType`` field and in Crossref's ``update-to`` relation, and was
simply never queried.

**Deduplication by title.** Dedup normalised the title
(``re.sub(r'[^a-zA-Z0-9]', '', title.lower())``) and compared for equality.
A bioRxiv preprint and its published version usually differ by a word or
two, so both survived — and then mutually "corroborated" a hypothesis while
being one piece of evidence. DOIs, and near-title matching for the
preprint/version case, fix that.

**No quality signal.** Every paper counted equally regardless of age, venue
or citation count, so a 1998 conference abstract weighed the same as a 2025
randomised trial.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+)\b")
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)
_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_PMC_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)


def extract_doi(paper: dict) -> str:
    """Pull a DOI from wherever it happens to live in a paper record."""
    for key in ("doi", "DOI", "url", "link", "id", "entry_id"):
        value = str(paper.get(key, ""))
        match = _DOI_RE.search(value)
        if match:
            return match.group(1).lower().rstrip(".").rstrip("/")
    return ""


def canonical_id(paper: dict) -> str:
    """Best available stable identifier, prefixed by scheme.

    Order matters: a DOI identifies the work, an arXiv ID identifies a
    preprint that may later acquire one, and a URL identifies nothing stable
    at all.
    """
    doi = extract_doi(paper)
    if doi:
        return f"doi:{doi}"

    blob = " ".join(str(paper.get(k, "")) for k in ("url", "id", "entry_id", "link"))
    for regex, scheme in ((_ARXIV_RE, "arxiv"), (_PMID_RE, "pmid"), (_PMC_RE, "pmc")):
        match = regex.search(blob)
        if match:
            return f"{scheme}:{match.group(1).lower()}"

    return f"title:{normalise_title(paper.get('title', ''))}"


def normalise_title(title: str) -> str:
    """Aggressive normalisation for near-duplicate detection."""
    text = re.sub(r"[^a-z0-9\s]", " ", (title or "").lower())
    # Drop stopwords that preprint/published pairs commonly gain or lose.
    stop = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
            "with", "by", "from", "via", "using", "novel", "new"}
    words = [w for w in text.split() if w and w not in stop]
    return " ".join(words)


def titles_are_near_duplicates(a: str, b: str, threshold: float = 0.92) -> bool:
    """Whether two titles are the same work under a different version."""
    na, nb = normalise_title(a), normalise_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@dataclass
class DedupReport:
    kept: list[dict] = field(default_factory=list)
    removed: list[tuple[dict, str]] = field(default_factory=list)

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    def render(self) -> str:
        if not self.removed:
            return f"Deduplication: {len(self.kept)} papers, no duplicates."
        lines = [f"Deduplication: {len(self.kept)} kept, {self.n_removed} removed."]
        for paper, reason in self.removed[:10]:
            lines.append(f"  - '{(paper.get('title') or '')[:60]}' — {reason}")
        return "\n".join(lines)


#: Preference order when two records describe the same work. A published
#: version supersedes its preprint.
_SOURCE_RANK = {
    "pubmed": 0, "europepmc": 1, "openalex": 2,
    "arxiv": 3, "biorxiv": 4, "medrxiv": 4,
}


def _preference(paper: dict) -> tuple:
    """Lower sorts better: prefer DOI-bearing, published, longer-abstract records."""
    has_doi = 0 if extract_doi(paper) else 1
    source_rank = _SOURCE_RANK.get(str(paper.get("source", "")).lower(), 9)
    abstract_len = -len(str(paper.get("summary", "")))
    return (has_doi, source_rank, abstract_len)


def deduplicate(papers: list[dict]) -> DedupReport:
    """Collapse duplicate records, preferring the most authoritative version."""
    report = DedupReport()
    ordered = sorted(papers, key=_preference)

    by_id: dict[str, dict] = {}
    for paper in ordered:
        cid = canonical_id(paper)

        if cid in by_id:
            report.removed.append((paper, f"duplicate identifier {cid}"))
            continue

        near = next(
            (
                kept for kept in by_id.values()
                if titles_are_near_duplicates(
                    paper.get("title", ""), kept.get("title", ""),
                )
            ),
            None,
        )
        if near is not None:
            report.removed.append((
                paper,
                f"near-duplicate of '{(near.get('title') or '')[:50]}' "
                f"(likely preprint/published pair)",
            ))
            continue

        by_id[cid] = paper

    report.kept = list(by_id.values())
    return report


# ---------------------------------------------------------------------------
# Retraction screening
# ---------------------------------------------------------------------------

@dataclass
class RetractionStatus:
    identifier: str = ""
    retracted: bool = False
    concern: bool = False       # expression of concern / correction
    reason: str = ""
    source: str = ""

    @property
    def usable(self) -> bool:
        return not self.retracted


#: PubMed publication types that mark a record as withdrawn or disputed.
_RETRACTION_TYPES = {
    "retracted publication", "retraction of publication",
    "withdrawn publication",
}
_CONCERN_TYPES = {
    "expression of concern", "published erratum", "corrected and republished article",
}


def _http_json(url: str, timeout: float = 6.0) -> dict | None:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NewAIScientist-LiteratureHygiene/1.0 (mailto:ai-scientist@example.com)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hygiene lookup failed for %s: %s", url, exc)
        return None


def check_retraction_crossref(doi: str, timeout: float = 6.0) -> RetractionStatus:
    """Query Crossref for retraction notices attached to a DOI.

    Crossref records retractions as an ``update-to`` relation of type
    ``retraction`` on the *notice*, and exposes ``updated-by`` on the
    retracted work itself.
    """
    status = RetractionStatus(identifier=doi, source="crossref")
    if not doi:
        return status

    data = _http_json(
        f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", timeout,
    )
    if not data:
        status.reason = "lookup unavailable"
        return status

    message = data.get("message", {})
    for update in message.get("updated-by", []) or []:
        label = str(update.get("type", "")).lower()
        if "retract" in label or "withdraw" in label:
            status.retracted = True
            status.reason = f"Crossref: {update.get('type')} ({update.get('DOI', '')})"
            return status
        if "concern" in label or "correct" in label or "erratum" in label:
            status.concern = True
            status.reason = f"Crossref: {update.get('type')}"

    if "retract" in str(message.get("title", [""])[0] if message.get("title") else "").lower():
        status.retracted = True
        status.reason = "Crossref: title indicates a retraction notice"

    return status


def check_retraction_pubmed(pmid: str, timeout: float = 6.0) -> RetractionStatus:
    """Query PubMed for retraction-related publication types."""
    status = RetractionStatus(identifier=pmid, source="pubmed")
    if not pmid:
        return status

    data = _http_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={urllib.parse.quote(pmid)}&retmode=json",
        timeout,
    )
    if not data:
        status.reason = "lookup unavailable"
        return status

    record = (data.get("result", {}) or {}).get(pmid, {})
    types = {str(t).lower() for t in record.get("pubtype", [])}

    if types & _RETRACTION_TYPES:
        status.retracted = True
        status.reason = f"PubMed publication type: {sorted(types & _RETRACTION_TYPES)}"
    elif types & _CONCERN_TYPES:
        status.concern = True
        status.reason = f"PubMed publication type: {sorted(types & _CONCERN_TYPES)}"

    return status


def screen_paper(paper: dict, timeout: float = 6.0) -> RetractionStatus:
    """Screen one paper via whichever identifier it carries."""
    doi = extract_doi(paper)
    if doi:
        status = check_retraction_crossref(doi, timeout)
        if status.retracted or status.concern:
            return status

    blob = " ".join(str(paper.get(k, "")) for k in ("url", "id", "link"))
    pmid_match = _PMID_RE.search(blob)
    if pmid_match:
        return check_retraction_pubmed(pmid_match.group(1), timeout)

    return RetractionStatus(identifier=canonical_id(paper), reason="no resolvable identifier")


async def screen_corpus(
    papers: list[dict],
    timeout: float = 6.0,
    max_concurrent: int = 4,
) -> tuple[list[dict], list[tuple[dict, RetractionStatus]]]:
    """Screen a corpus concurrently. Returns ``(clean, excluded)``.

    Papers under an expression of concern are kept but annotated, so the
    reviewer sees the flag; retracted papers are removed outright.
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _screen(paper: dict) -> tuple[dict, RetractionStatus]:
        async with semaphore:
            status = await asyncio.to_thread(screen_paper, paper, timeout)
            return paper, status

    outcomes = await asyncio.gather(
        *[_screen(p) for p in papers], return_exceptions=True,
    )

    clean: list[dict] = []
    excluded: list[tuple[dict, RetractionStatus]] = []

    for outcome in outcomes:
        if isinstance(outcome, Exception):
            logger.debug("Screening raised: %s", outcome)
            continue
        paper, status = outcome
        if status.retracted:
            excluded.append((paper, status))
            logger.warning(
                "EXCLUDED retracted paper: '%s' — %s",
                (paper.get("title") or "")[:60], status.reason,
            )
            continue
        if status.concern:
            paper = dict(paper)
            paper["integrity_flag"] = status.reason
        clean.append(paper)

    return clean, excluded


# ---------------------------------------------------------------------------
# Quality weighting
# ---------------------------------------------------------------------------

def parse_year(paper: dict) -> int | None:
    for key in ("published", "year", "date", "publication_date"):
        match = re.search(r"(19|20)\d{2}", str(paper.get(key, "")))
        if match:
            return int(match.group(0))
    return None


def quality_weight(
    paper: dict,
    now_year: int | None = None,
    half_life_years: float = 8.0,
) -> float:
    """Heuristic weight in roughly [0.2, 1.5] for evidence weighting.

    Deliberately gentle: an old paper is downweighted, not discarded, and a
    preprint is downweighted, not dismissed. The point is to stop treating a
    1998 abstract and a 2025 trial as interchangeable, not to enforce a
    citation hierarchy.
    """
    weight = 1.0
    now_year = now_year or datetime.now().year

    year = parse_year(paper)
    if year:
        age = max(0, now_year - year)
        weight *= 0.5 ** (age / half_life_years) * 0.5 + 0.5   # floor at 0.5×

    source = str(paper.get("source", "")).lower()
    if source in ("biorxiv", "medrxiv", "arxiv"):
        weight *= 0.85          # not peer reviewed
    elif source in ("pubmed", "europepmc"):
        weight *= 1.1

    if paper.get("integrity_flag"):
        weight *= 0.4           # expression of concern

    # Citation signal. Semantic Scholar supplies these; arXiv and the PubMed
    # E-utilities do not, which is why this branch was previously dead.
    # influentialCitationCount is preferred where available: it excludes
    # perfunctory citations and is a better quality proxy than raw counts.
    try:
        influential = int(paper.get("influential_citation_count", 0) or 0)
        citations = int(paper.get("citation_count", 0) or 0)
    except (TypeError, ValueError):
        influential = citations = 0

    import math

    if influential > 0:
        weight *= 1.0 + min(0.35, math.log10(1 + influential) / 6.0)
    elif citations > 0:
        weight *= 1.0 + min(0.3, math.log10(1 + citations) / 10.0)

    # Evidence hierarchy from S2 publication types: a meta-analysis and a
    # case report are not interchangeable evidence.
    evidence = paper.get("evidence_weight")
    if isinstance(evidence, (int, float)) and evidence > 0:
        weight *= float(evidence)

    return round(max(0.2, min(2.0, weight)), 3)


def apply_hygiene(papers: list[dict]) -> tuple[list[dict], DedupReport]:
    """Deduplicate and attach quality weights. Synchronous; no network.

    Retraction screening is separate (``screen_corpus``) because it hits the
    network and callers may want to run it only on papers that actually make
    it into the RAG index.
    """
    report = deduplicate(papers)
    for paper in report.kept:
        paper["quality_weight"] = quality_weight(paper)
        paper["canonical_id"] = canonical_id(paper)
    report.kept.sort(key=lambda p: p.get("quality_weight", 0.0), reverse=True)
    return report.kept, report


__all__ = [
    "DedupReport",
    "RetractionStatus",
    "apply_hygiene",
    "canonical_id",
    "check_retraction_crossref",
    "check_retraction_pubmed",
    "deduplicate",
    "extract_doi",
    "normalise_title",
    "parse_year",
    "quality_weight",
    "screen_corpus",
    "screen_paper",
    "titles_are_near_duplicates",
]
