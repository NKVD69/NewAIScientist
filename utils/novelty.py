"""
utils/novelty.py — Grounded novelty assessment via prior-art search.

Replaces the fabricated novelty score. Two paths existed, both broken.

**Fallback path** (``ReflectionAgent._assess_novelty``)::

    score = 0.6
    if "simulated" in hypothesis.generation_method:  score = 0.55
    elif "llm" in hypothesis.generation_method:      score = 0.75
    elif hypothesis.generation_method == "evolved":  score = 0.65

Novelty was a function of *how the hypothesis had been generated*. A
hypothesis scored 36% more novel for having come from an LLM rather than the
simulation stub. That is an artefact of plumbing promoted to a scientific
measurement — and it propagated into the Bradley-Terry prior (novelty carries
weight 0.25) and from there into which hypothesis got written up.

**LLM path.** The meta-reviewer returned ``novelty_score`` introspectively,
with no retrieval, for a domain where parametric memory is weakest: recent
and niche literature.

What replaces them is a search. Novelty is not something a model can
introspect; it is a claim about what does and does not exist in the
literature, and the only way to assess it is to look.

The output is deliberately **auditable**. A scalar of 0.75 is worthless to a
reviewer; the three nearest papers, with titles and links, let a human judge
for themselves. That is the point of the report.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PriorArtHit:
    """One paper that may already make the hypothesis's claim."""

    title: str = ""
    year: int | None = None
    venue: str = ""
    url: str = ""
    citation_count: int = 0
    similarity: float = 0.0
    source: str = "semanticscholar"
    tldr: str = ""

    def render(self) -> str:
        bits = [f"“{self.title[:90]}”"]
        if self.year:
            bits.append(str(self.year))
        if self.venue:
            bits.append(self.venue[:40])
        if self.citation_count:
            bits.append(f"{self.citation_count} citations")
        return " · ".join(bits)


@dataclass
class NoveltyReport:
    """Auditable novelty assessment."""

    hypothesis_title: str = ""
    query: str = ""
    corpus_distance: float = 1.0        # 1 - max similarity to known work
    prior_art: list[PriorArtHit] = field(default_factory=list)
    edge_is_new: bool = True            # relation absent from the knowledge graph
    searched: bool = False              # whether a real search ran
    error: str = ""
    #: "embedding" (cosine) or "token" (Jaccard). These live on different
    #: scales -- Jaccard runs far lower for the same semantic closeness -- so
    #: every threshold below must be read against the right one. Hard-coding a
    #: cosine-calibrated 0.85 meant the token fallback never flagged anything.
    similarity_method: str = "token"

    #: Similarity above which a hit counts as probable prior art, per method.
    PRIOR_ART_THRESHOLD = {"embedding": 0.85, "token": 0.45}
    #: Similarity above which the claim is treated as already made.
    RESTATEMENT_THRESHOLD = {"embedding": 0.90, "token": 0.60}

    @property
    def prior_art_threshold(self) -> float:
        return self.PRIOR_ART_THRESHOLD.get(self.similarity_method, 0.85)

    @property
    def restatement_threshold(self) -> float:
        return self.RESTATEMENT_THRESHOLD.get(self.similarity_method, 0.90)

    @property
    def probable_prior_art(self) -> list[PriorArtHit]:
        """Hits close enough to warrant a human look, on the right scale."""
        return [h for h in self.prior_art if h.similarity >= self.prior_art_threshold]

    @property
    def score(self) -> float:
        """Novelty in [0, 1], from evidence rather than generation method.

        When no search could run the score is 0.5 — explicit ignorance — and
        ``searched`` is False so callers can tell "we do not know" apart from
        "we checked and it is middling". The old code could not express that
        distinction, which is how a plumbing artefact passed for a measurement.
        """
        if not self.searched:
            return 0.5
        score = self.corpus_distance
        if not self.edge_is_new:
            score *= 0.6
        if self.prior_art and self.prior_art[0].similarity >= self.restatement_threshold:
            score = min(score, 0.15)
        return round(max(0.0, min(1.0, score)), 3)

    @property
    def level(self) -> str:
        score = self.score
        if not self.searched:
            return "unknown"
        if score >= 0.8:
            return "very_high"
        if score >= 0.6:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def render(self) -> str:
        if not self.searched:
            return (
                f"Novelty: NOT ASSESSED ({self.error or 'no prior-art search available'}). "
                "Treat the score as unknown, not as evidence of novelty."
            )

        lines = [
            f"## Novelty assessment — {self.level} ({self.score:.2f})",
            "",
            f"Query: {self.query}",
            f"Distance from nearest known work: {self.corpus_distance:.2f} "
            f"(similarity via {self.similarity_method})",
        ]
        if not self.edge_is_new:
            lines.append(
                "⚠ The claimed relation already exists in the knowledge graph."
            )

        if self.prior_art:
            lines += ["", "Nearest prior art (judge for yourself):"]
            for hit in self.prior_art[:5]:
                lines.append(f"  {hit.similarity:.2f}  {hit.render()}")
                if hit.url:
                    lines.append(f"        {hit.url}")
                if hit.tldr:
                    lines.append(f"        TL;DR: {hit.tldr[:140]}")
        else:
            lines += ["", "No closely related work surfaced by the prior-art search."]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "level": self.level,
            "searched": self.searched,
            "corpus_distance": round(self.corpus_distance, 4),
            "similarity_method": self.similarity_method,
            "edge_is_new": self.edge_is_new,
            "query": self.query,
            "error": self.error,
            "prior_art": [
                {
                    "title": h.title, "year": h.year, "venue": h.venue,
                    "url": h.url, "similarity": round(h.similarity, 4),
                    "citation_count": h.citation_count,
                }
                for h in self.prior_art
            ],
        }


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with", "by",
    "from", "via", "using", "that", "this", "is", "are", "we", "our", "may",
    "can", "could", "novel", "new", "study", "investigate", "hypothesis",
    "propose", "proposed", "suggests", "role", "effect", "effects",
})


def build_prior_art_query(hypothesis, max_terms: int = 12) -> str:
    """Build a search query from the hypothesis's substance, not its title.

    Titles carry framing ("A Novel Mechanism for…") that a semantic search
    engine will happily match against other papers' framing. The mechanism
    and predictions carry the actual claim, which is what needs checking.
    """
    parts = [
        getattr(hypothesis, "title", "") or "",
        getattr(hypothesis, "mechanism", "") or "",
        " ".join((getattr(hypothesis, "testable_predictions", None) or [])[:2]),
    ]
    text = " ".join(parts)

    # Keep entity-like tokens: capitalised names, gene symbols, compounds,
    # alphanumerics — these carry the claim's specificity.
    entities = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b|\b[A-Za-z]+\d+[A-Za-z0-9\-]*\b", text)

    words = [
        w for w in re.findall(r"\b[a-z]{4,}\b", text.lower())
        if w not in _STOPWORDS
    ]

    seen: set[str] = set()
    terms: list[str] = []
    for term in [*entities, *words]:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            terms.append(term)
        if len(terms) >= max_terms:
            break

    return " ".join(terms) or (getattr(hypothesis, "title", "") or "")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _token_similarity(a: str, b: str) -> float:
    """Jaccard over content words. Fallback when no embedder is available."""
    ta = {w for w in re.findall(r"\b[a-z]{3,}\b", (a or "").lower()) if w not in _STOPWORDS}
    tb = {w for w in re.findall(r"\b[a-z]{3,}\b", (b or "").lower()) if w not in _STOPWORDS}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _embedding_similarity(model, a: str, b_list: list[str]) -> list[float]:
    """Cosine similarity via a sentence-transformer, when one is loaded."""
    import numpy as np

    vectors = model.encode([a, *b_list], normalize_embeddings=True)
    query = np.asarray(vectors[0])
    return [float(np.dot(query, np.asarray(v))) for v in vectors[1:]]


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

async def assess_novelty(
    hypothesis,
    rag_engine=None,
    s2_client=None,
    graph_agent=None,
    top_k: int = 10,
    timeout: float = 25.0,
) -> NoveltyReport:
    """Assess novelty by triangulating three independent signals.

    1. **Prior art** — a semantic search of Semantic Scholar's corpus. This
       is the load-bearing signal: it looks outside whatever the local RAG
       index happens to contain.
    2. **Local corpus distance** — similarity to the indexed papers, which
       catches the case where the hypothesis restates something the system
       itself retrieved earlier in the run.
    3. **Graph edge** — whether the claimed relation is already an edge in
       the knowledge graph.

    Degrades gracefully at every step. Crucially, a failed search yields
    ``searched=False`` rather than a confident-looking default: the failure
    mode being fixed here is precisely a number that looked like a
    measurement and was not.
    """
    report = NoveltyReport(
        hypothesis_title=getattr(hypothesis, "title", "") or "",
        query=build_prior_art_query(hypothesis),
    )

    # --- 1. Prior art via Semantic Scholar -----------------------------
    hits: list[PriorArtHit] = []
    try:
        if s2_client is None:
            from utils.semantic_scholar import get_client
            s2_client = get_client()

        papers = await asyncio.wait_for(
            s2_client.search(report.query, limit=top_k), timeout=timeout,
        )
        claim_text = " ".join([
            report.hypothesis_title,
            getattr(hypothesis, "mechanism", "") or "",
        ])

        candidates = [f"{p.title} {p.abstract or p.tldr}" for p in papers]
        similarities: list[float] = []
        embedder = getattr(rag_engine, "embedding_model", None) if rag_engine else None
        if embedder is not None and candidates:
            try:
                similarities = _embedding_similarity(embedder, claim_text, candidates)
                report.similarity_method = "embedding"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Embedding similarity failed (%s) — using tokens.", exc)
        if not similarities:
            similarities = [_token_similarity(claim_text, c) for c in candidates]
            report.similarity_method = "token"

        for paper, similarity in zip(papers, similarities, strict=True):
            hits.append(PriorArtHit(
                title=paper.title,
                year=paper.year,
                venue=paper.venue,
                url=paper.url,
                citation_count=paper.citation_count,
                similarity=max(0.0, min(1.0, similarity)),
                tldr=paper.tldr,
            ))

        hits.sort(key=lambda h: h.similarity, reverse=True)
        report.prior_art = hits
        report.searched = True

    except TimeoutError:
        report.error = "prior-art search timed out"
        logger.warning("Novelty: S2 search timed out for '%s'.", report.hypothesis_title[:40])
    except Exception as exc:  # noqa: BLE001
        report.error = f"prior-art search failed: {exc}"
        logger.warning("Novelty: %s", report.error)

    # --- 2. Distance from the locally indexed corpus --------------------
    local_max = 0.0
    if rag_engine is not None:
        try:
            chunks = await _query_rag(rag_engine, report.query, top_k=10)
            claim_text = report.hypothesis_title + " " + (
                getattr(hypothesis, "mechanism", "") or ""
            )
            for chunk in chunks or []:
                score = chunk.get("score")
                if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                    local_max = max(local_max, float(score))
                else:
                    local_max = max(
                        local_max, _token_similarity(claim_text, chunk.get("text", "")),
                    )
            report.searched = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Novelty: local corpus check failed: %s", exc)

    external_max = hits[0].similarity if hits else 0.0
    report.corpus_distance = 1.0 - max(external_max, local_max)

    # --- 3. Knowledge-graph edge ---------------------------------------
    if graph_agent is not None:
        try:
            checker = getattr(graph_agent, "has_edge", None)
            subject = getattr(hypothesis, "subject", None)
            predicate = getattr(hypothesis, "predicate", None)
            obj = getattr(hypothesis, "object", None)
            if callable(checker) and subject and obj:
                report.edge_is_new = not checker(subject, predicate, obj)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Novelty: graph edge check failed: %s", exc)

    return report


async def _query_rag(rag_engine, query: str, top_k: int):
    """Query the RAG index by whichever interface is available."""
    if hasattr(rag_engine, "query_hybrid") and getattr(rag_engine, "enable_hybrid", False):
        try:
            return await rag_engine.query_hybrid(query, top_k=top_k)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(rag_engine, "query"):
        return await rag_engine.query(query, top_k)
    return []


async def assess_many(
    hypotheses: list,
    rag_engine=None,
    s2_client=None,
    graph_agent=None,
    max_concurrent: int = 3,
) -> dict[str, NoveltyReport]:
    """Assess a batch, bounded so the S2 rate limiter is not overwhelmed."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(hypothesis):
        async with semaphore:
            return hypothesis.id, await assess_novelty(
                hypothesis, rag_engine=rag_engine,
                s2_client=s2_client, graph_agent=graph_agent,
            )

    outcomes = await asyncio.gather(
        *[_one(h) for h in hypotheses], return_exceptions=True,
    )
    reports: dict[str, NoveltyReport] = {}
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            logger.debug("Novelty assessment raised: %s", outcome)
            continue
        hypothesis_id, report = outcome
        reports[hypothesis_id] = report
    return reports


def apply_report(hypothesis, report: NoveltyReport) -> None:
    """Write a novelty report onto a hypothesis.

    Sets ``novelty_level`` and stores the full auditable report. When the
    search did not run, ``novelty_level`` becomes ``"unknown"`` rather than a
    plausible-looking value — the whole point of this module.
    """
    hypothesis.novelty_level = report.level
    hypothesis.novelty_report = report.to_dict()

    if report.searched and report.probable_prior_art:
        nearest = report.probable_prior_art[0]
        hypothesis.limitations.append(
            f"Possible prior art (similarity {nearest.similarity:.2f}): "
            f"“{nearest.title[:80]}”"
            + (f" — {nearest.url}" if nearest.url else "")
        )


__all__ = [
    "NoveltyReport",
    "PriorArtHit",
    "apply_report",
    "assess_many",
    "assess_novelty",
    "build_prior_art_query",
]
