"""
utils/semantic_scholar.py — Semantic Scholar Graph API client.

Semantic Scholar fills three gaps the existing sources cannot.

**Citation counts.** ``literature_hygiene.quality_weight`` reads a
``citation_count`` field that nothing ever populated — arXiv and the PubMed
E-utilities do not return one. S2 does, plus ``influentialCitationCount``,
which is a better signal than raw counts because it excludes perfunctory
citations.

**Publication type.** S2 labels records as ``MetaAnalysis``,
``ClinicalTrial``, ``Review``, ``CaseReport`` and so on. That is an evidence
hierarchy, and treating a meta-analysis and a case report as interchangeable
evidence — which the system did — is a real error in a biomedical setting.

**Grounded novelty.** The audit found ``novelty_score`` was, in the fallback
path, a function of *how the hypothesis had been generated*
(``0.75 if "llm" in generation_method``) and, in the LLM path, an
introspective guess with no retrieval. S2's search is semantic rather than
lexical, so it can actually answer "has someone already claimed this?" See
``utils.novelty``.

API notes
---------
* Base: ``https://api.semanticscholar.org/graph/v1``
* IDs accept prefixes: ``DOI:``, ``ARXIV:``, ``PMID:``, ``PMCID:``,
  ``CorpusId:``, or a bare 40-char S2 hash.
* Unauthenticated traffic shares a small pool (~100 requests / 5 min) and
  429s readily. With a key (``S2_API_KEY`` / ``SEMANTIC_SCHOLAR_API_KEY``)
  the documented rate is 1 req/s. The client rate-limits itself accordingly
  and backs off on 429 rather than hammering.

.. warning::
   This module was written against the documented API contract but could not
   be exercised against the live service in the development sandbox
   (``api.semanticscholar.org`` was not reachable). Endpoint shapes and field
   names follow the published schema; run ``python -m utils.semantic_scholar
   --selftest`` against the real API before relying on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE = "https://api.semanticscholar.org/recommendations/v1"

#: Fields requested by default. Kept explicit: S2 returns only paperId unless
#: fields are named, and over-requesting slows every call.
DEFAULT_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,publicationDate,"
    "referenceCount,citationCount,influentialCitationCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,publicationTypes,journal,authors,tldr"
)

#: Evidence hierarchy. A meta-analysis and a case report are not the same
#: kind of claim, and weighting them equally is a substantive error.
PUBLICATION_TYPE_WEIGHT: dict[str, float] = {
    "MetaAnalysis": 1.40,
    "ClinicalTrial": 1.30,
    "Review": 1.10,
    "JournalArticle": 1.00,
    "Study": 1.00,
    "Conference": 0.90,
    "Dataset": 0.85,
    "Book": 0.85,
    "BookSection": 0.80,
    "CaseReport": 0.70,
    "LettersAndComments": 0.50,
    "Editorial": 0.45,
    "News": 0.30,
}


class SemanticScholarError(RuntimeError):
    """Raised for unrecoverable API failures."""


class RateLimited(SemanticScholarError):
    """Raised when the API returns 429 after exhausting retries."""


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple async token bucket.

    S2 is strict and returns 429 readily on the shared unauthenticated pool.
    Self-limiting is cheaper than retry storms and is the difference between
    a working integration and one that mostly returns errors.
    """

    def __init__(self, rate_per_second: float):
        self.min_interval = 1.0 / max(rate_per_second, 0.01)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class S2Paper:
    """A Semantic Scholar record, normalised to this codebase's paper dict."""

    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    is_open_access: bool = False
    pdf_url: str = ""
    fields_of_study: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    tldr: str = ""
    url: str = ""

    @classmethod
    def from_api(cls, raw: dict) -> S2Paper:
        external = raw.get("externalIds") or {}
        oa = raw.get("openAccessPdf") or {}
        tldr = raw.get("tldr") or {}
        return cls(
            paper_id=raw.get("paperId", "") or "",
            title=raw.get("title", "") or "",
            abstract=raw.get("abstract", "") or "",
            year=raw.get("year"),
            venue=(raw.get("venue") or (raw.get("journal") or {}).get("name") or ""),
            authors=[a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")],
            doi=(external.get("DOI") or "").lower(),
            arxiv_id=external.get("ArXiv", "") or "",
            pmid=str(external.get("PubMed", "") or ""),
            citation_count=int(raw.get("citationCount") or 0),
            influential_citation_count=int(raw.get("influentialCitationCount") or 0),
            reference_count=int(raw.get("referenceCount") or 0),
            is_open_access=bool(raw.get("isOpenAccess")),
            pdf_url=oa.get("url", "") or "",
            fields_of_study=list(raw.get("fieldsOfStudy") or []),
            publication_types=list(raw.get("publicationTypes") or []),
            tldr=tldr.get("text", "") or "",
            url=raw.get("url", "") or "",
        )

    @property
    def evidence_weight(self) -> float:
        """Weight from publication type — the evidence hierarchy."""
        if not self.publication_types:
            return 1.0
        return max(
            PUBLICATION_TYPE_WEIGHT.get(t, 1.0) for t in self.publication_types
        )

    def to_paper_dict(self) -> dict:
        """Convert to the paper dict shape used throughout this codebase.

        Field names mirror the arXiv/PubMed adapters so the record flows
        through ``literature_hygiene``, the chunker and the RAG index with no
        special-casing.
        """
        return {
            "title": self.title,
            "summary": self.abstract or self.tldr,
            "authors": self.authors,
            "url": self.url or (f"https://doi.org/{self.doi}" if self.doi else ""),
            "pdf_url": self.pdf_url,
            "published": str(self.year) if self.year else "",
            "source": "semanticscholar",
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            # Populates literature_hygiene.quality_weight, which read this
            # field and never received one from arXiv or PubMed.
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "venue": self.venue,
            "publication_types": self.publication_types,
            "evidence_weight": self.evidence_weight,
            "fields_of_study": self.fields_of_study,
            "s2_paper_id": self.paper_id,
            "tldr": self.tldr,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SemanticScholarClient:
    """Async wrapper over the Graph and Recommendations APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        rate_per_second: float | None = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("S2_API_KEY")
            or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            or ""
        ).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        # Unauthenticated traffic shares a small pool; go deliberately slow.
        default_rate = 1.0 if self.api_key else 0.34
        self._limiter = _RateLimiter(rate_per_second or default_rate)
        self.requests_made = 0
        self.rate_limit_hits = 0

    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        headers = {
            "User-Agent": "NewAIScientist/3.1 (research assistant; +https://github.com/NKVD69/NewAIScientist)",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _request(
        self,
        url: str,
        method: str = "GET",
        payload: dict | None = None,
    ) -> dict | list | None:
        """Issue one API call with rate limiting and 429/5xx backoff."""
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(1, self.max_retries + 1):
            await self._limiter.acquire()

            def _do():
                request = urllib.request.Request(
                    url, data=body, method=method, headers=headers,
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            try:
                self.requests_made += 1
                return await asyncio.to_thread(_do)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    self.rate_limit_hits += 1
                    backoff = min(30.0, 2.0 ** attempt)
                    logger.warning(
                        "Semantic Scholar rate limit (attempt %d/%d), waiting %.0fs. "
                        "%s",
                        attempt, self.max_retries, backoff,
                        "Set S2_API_KEY for a higher quota."
                        if not self.api_key else "",
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(backoff)
                        continue
                    raise RateLimited("Semantic Scholar rate limit exhausted") from exc
                if exc.code == 404:
                    return None
                if 500 <= exc.code < 600 and attempt < self.max_retries:
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                logger.warning("Semantic Scholar HTTP %s for %s", exc.code, url)
                return None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                logger.debug("Semantic Scholar request failed (%s): %s", attempt, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                return None
        return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 20,
        fields: str = DEFAULT_FIELDS,
        year: str | None = None,
        min_citation_count: int | None = None,
        publication_types: list[str] | None = None,
        open_access_only: bool = False,
        fields_of_study: list[str] | None = None,
    ) -> list[S2Paper]:
        """Relevance search over the corpus.

        ``year`` accepts the API's range syntax (``"2020-"``, ``"2018-2022"``).
        Filters matter here: the existing PubMed adapter hard-codes
        ``AND "free full text"[Filter]``, which biases the corpus toward
        particular publishers; S2 lets open access be an explicit, reversible
        choice rather than an invisible constraint.
        """
        params = {
            "query": query,
            "limit": str(min(max(1, limit), 100)),
            "fields": fields,
        }
        if year:
            params["year"] = year
        if min_citation_count is not None:
            params["minCitationCount"] = str(min_citation_count)
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if open_access_only:
            params["openAccessPdf"] = ""
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        url = f"{GRAPH_BASE}/paper/search?{urllib.parse.urlencode(params)}"
        data = await self._request(url)
        if not isinstance(data, dict):
            return []
        return [S2Paper.from_api(item) for item in data.get("data", []) or []]

    async def search_bulk(
        self,
        query: str,
        limit: int = 200,
        fields: str = DEFAULT_FIELDS,
        year: str | None = None,
    ) -> list[S2Paper]:
        """Bulk search with token pagination.

        Supports boolean syntax (``"KRAS + (inhibitor | degrader)"``) and
        returns up to 1000 per page. Used when a broad corpus matters more
        than precise relevance ranking — the previous ceiling of 10 papers
        per goal was two orders of magnitude short for a novelty claim.
        """
        collected: list[S2Paper] = []
        token: str | None = None

        while len(collected) < limit:
            params = {"query": query, "fields": fields}
            if year:
                params["year"] = year
            if token:
                params["token"] = token

            url = f"{GRAPH_BASE}/paper/search/bulk?{urllib.parse.urlencode(params)}"
            data = await self._request(url)
            if not isinstance(data, dict):
                break

            batch = data.get("data") or []
            collected.extend(S2Paper.from_api(item) for item in batch)

            token = data.get("token")
            if not token or not batch:
                break

        return collected[:limit]

    async def match_title(self, title: str, fields: str = DEFAULT_FIELDS) -> S2Paper | None:
        """Find the single closest title match. Useful for citation resolution."""
        params = {"query": title, "fields": fields}
        url = f"{GRAPH_BASE}/paper/search/match?{urllib.parse.urlencode(params)}"
        data = await self._request(url)
        if not isinstance(data, dict):
            return None
        matches = data.get("data") or []
        return S2Paper.from_api(matches[0]) if matches else None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get_paper(self, paper_id: str, fields: str = DEFAULT_FIELDS) -> S2Paper | None:
        """Fetch one paper. ``paper_id`` may be ``DOI:...``, ``ARXIV:...`` etc."""
        params = urllib.parse.urlencode({"fields": fields})
        url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id, safe=':')}?{params}"
        data = await self._request(url)
        return S2Paper.from_api(data) if isinstance(data, dict) else None

    async def get_papers_batch(
        self,
        paper_ids: list[str],
        fields: str = DEFAULT_FIELDS,
    ) -> list[S2Paper | None]:
        """Fetch up to 500 papers in one call.

        Batching is what makes enriching an existing corpus affordable: one
        request for a whole arXiv/PubMed result set instead of N.
        """
        results: list[S2Paper | None] = []
        for start in range(0, len(paper_ids), 500):
            chunk = paper_ids[start:start + 500]
            url = f"{GRAPH_BASE}/paper/batch?{urllib.parse.urlencode({'fields': fields})}"
            data = await self._request(url, method="POST", payload={"ids": chunk})
            if not isinstance(data, list):
                results.extend([None] * len(chunk))
                continue
            results.extend(
                S2Paper.from_api(item) if isinstance(item, dict) else None
                for item in data
            )
        return results

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def get_references(
        self, paper_id: str, limit: int = 100, fields: str = DEFAULT_FIELDS,
    ) -> list[S2Paper]:
        """Papers this one cites — the intellectual ancestry of a claim."""
        params = urllib.parse.urlencode({"fields": fields, "limit": str(min(limit, 1000))})
        url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id, safe=':')}/references?{params}"
        data = await self._request(url)
        if not isinstance(data, dict):
            return []
        return [
            S2Paper.from_api(item["citedPaper"])
            for item in data.get("data", []) or []
            if isinstance(item.get("citedPaper"), dict)
        ]

    async def get_citations(
        self, paper_id: str, limit: int = 100, fields: str = DEFAULT_FIELDS,
    ) -> list[S2Paper]:
        """Papers citing this one — including anything that refutes it."""
        params = urllib.parse.urlencode({"fields": fields, "limit": str(min(limit, 1000))})
        url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id, safe=':')}/citations?{params}"
        data = await self._request(url)
        if not isinstance(data, dict):
            return []
        return [
            S2Paper.from_api(item["citingPaper"])
            for item in data.get("data", []) or []
            if isinstance(item.get("citingPaper"), dict)
        ]

    async def recommend(
        self, paper_id: str, limit: int = 20, fields: str = DEFAULT_FIELDS,
    ) -> list[S2Paper]:
        """Papers S2 considers related. A second angle on prior art."""
        params = urllib.parse.urlencode({"fields": fields, "limit": str(limit)})
        url = (
            f"{RECOMMENDATIONS_BASE}/papers/forpaper/"
            f"{urllib.parse.quote(paper_id, safe=':')}?{params}"
        )
        data = await self._request(url)
        if not isinstance(data, dict):
            return []
        return [S2Paper.from_api(item) for item in data.get("recommendedPapers", []) or []]

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    async def enrich_papers(self, papers: list[dict]) -> list[dict]:
        """Add S2 metadata to papers already retrieved from arXiv/PubMed.

        Resolves each record by DOI, arXiv ID or PMID in a single batch call,
        then fills in citation counts, publication types and open-access PDF
        links. Records that cannot be resolved pass through untouched — this
        must never drop papers.
        """
        if not papers:
            return papers

        index: dict[int, str] = {}
        for position, paper in enumerate(papers):
            from utils.literature_hygiene import extract_doi

            doi = extract_doi(paper)
            if doi:
                index[position] = f"DOI:{doi}"
                continue
            blob = " ".join(str(paper.get(k, "")) for k in ("url", "id", "entry_id"))
            import re

            arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", blob, re.I)
            if arxiv:
                index[position] = f"ARXIV:{arxiv.group(1)}"
                continue
            pmid = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", blob, re.I)
            if pmid:
                index[position] = f"PMID:{pmid.group(1)}"

        if not index:
            logger.info("No resolvable identifiers — skipping S2 enrichment.")
            return papers

        positions = list(index)
        try:
            fetched = await self.get_papers_batch([index[p] for p in positions])
        except SemanticScholarError as exc:
            logger.warning("S2 enrichment unavailable (%s) — corpus unchanged.", exc)
            return papers

        enriched = 0
        for position, s2 in zip(positions, fetched, strict=True):
            if s2 is None:
                continue
            paper = papers[position]
            paper["citation_count"] = s2.citation_count
            paper["influential_citation_count"] = s2.influential_citation_count
            paper["publication_types"] = s2.publication_types
            paper["evidence_weight"] = s2.evidence_weight
            paper["s2_paper_id"] = s2.paper_id
            if s2.venue and not paper.get("venue"):
                paper["venue"] = s2.venue
            if s2.pdf_url and not paper.get("pdf_url"):
                paper["pdf_url"] = s2.pdf_url
            if s2.doi and not paper.get("doi"):
                paper["doi"] = s2.doi
            if s2.tldr and not paper.get("tldr"):
                paper["tldr"] = s2.tldr
            enriched += 1

        logger.info("S2 enrichment: %d/%d papers matched.", enriched, len(papers))
        return papers

    def stats(self) -> dict:
        return {
            "requests_made": self.requests_made,
            "rate_limit_hits": self.rate_limit_hits,
            "authenticated": bool(self.api_key),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_CLIENT: SemanticScholarClient | None = None


def get_client() -> SemanticScholarClient:
    """Return the shared client, so the rate limiter is actually shared."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SemanticScholarClient()
    return _CLIENT


async def search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    """Search S2 and return paper dicts in this codebase's shape."""
    papers = await get_client().search(query, limit=max_results)
    return [p.to_paper_dict() for p in papers]


__all__ = [
    "DEFAULT_FIELDS",
    "PUBLICATION_TYPE_WEIGHT",
    "RateLimited",
    "S2Paper",
    "SemanticScholarClient",
    "SemanticScholarError",
    "get_client",
    "search_semantic_scholar",
]
