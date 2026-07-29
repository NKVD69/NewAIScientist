"""
Tests for the Semantic Scholar client and grounded novelty assessment.

The live API is not reachable from the test environment, so every network
call is mocked at the ``_request`` boundary. That covers response parsing,
identifier handling, rate-limit behaviour and graceful degradation — but not
the real API contract. Run the integration check against the live service
before trusting this in production.
"""

from __future__ import annotations

import asyncio

import pytest

from models.hypothesis import Hypothesis
from utils import novelty
from utils.literature_hygiene import quality_weight
from utils.semantic_scholar import (
    PUBLICATION_TYPE_WEIGHT,
    RateLimited,
    S2Paper,
    SemanticScholarClient,
)

RAW_PAPER = {
    "paperId": "abc123",
    "title": "KRAS G12C inhibition in non-small-cell lung cancer",
    "abstract": "We report covalent inhibition of KRAS G12C.",
    "year": 2024,
    "venue": "Nature",
    "externalIds": {"DOI": "10.1038/S41586-024-0001", "PubMed": "38123456",
                    "ArXiv": "2401.01234"},
    "citationCount": 412,
    "influentialCitationCount": 38,
    "referenceCount": 61,
    "isOpenAccess": True,
    "openAccessPdf": {"url": "https://example.org/paper.pdf"},
    "fieldsOfStudy": ["Medicine", "Biology"],
    "publicationTypes": ["JournalArticle", "MetaAnalysis"],
    "authors": [{"name": "A. Smith"}, {"name": "B. Jones"}],
    "tldr": {"text": "Covalent KRAS G12C inhibitors shrink tumours."},
    "url": "https://www.semanticscholar.org/paper/abc123",
}


def _client_returning(payload, calls: list | None = None) -> SemanticScholarClient:
    """Client whose HTTP layer is replaced by a canned response."""
    client = SemanticScholarClient(api_key="test-key")

    async def fake_request(url, method="GET", payload_body=None, **kwargs):
        if calls is not None:
            calls.append((url, method))
        return payload(url) if callable(payload) else payload

    client._request = fake_request  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestS2Paper:
    def test_parses_every_field(self):
        p = S2Paper.from_api(RAW_PAPER)
        assert p.paper_id == "abc123"
        assert p.citation_count == 412
        assert p.influential_citation_count == 38
        assert p.authors == ["A. Smith", "B. Jones"]
        assert p.tldr.startswith("Covalent")
        assert p.pdf_url.endswith(".pdf")

    def test_doi_is_lowercased(self):
        assert S2Paper.from_api(RAW_PAPER).doi == "10.1038/s41586-024-0001"

    def test_missing_fields_do_not_raise(self):
        p = S2Paper.from_api({"paperId": "x"})
        assert p.title == "" and p.citation_count == 0 and p.authors == []

    def test_null_nested_objects_are_tolerated(self):
        """S2 returns explicit nulls for tldr/openAccessPdf, not omissions."""
        p = S2Paper.from_api({"paperId": "x", "tldr": None, "openAccessPdf": None,
                              "externalIds": None, "authors": None})
        assert p.tldr == "" and p.pdf_url == "" and p.doi == ""

    def test_venue_falls_back_to_journal_name(self):
        p = S2Paper.from_api({"paperId": "x", "venue": "", "journal": {"name": "Cell"}})
        assert p.venue == "Cell"

    def test_evidence_hierarchy_takes_the_strongest_type(self):
        p = S2Paper.from_api(RAW_PAPER)   # JournalArticle + MetaAnalysis
        assert p.evidence_weight == PUBLICATION_TYPE_WEIGHT["MetaAnalysis"]

    def test_case_report_is_weaker_than_meta_analysis(self):
        assert (PUBLICATION_TYPE_WEIGHT["CaseReport"]
                < PUBLICATION_TYPE_WEIGHT["MetaAnalysis"])

    def test_unknown_publication_type_is_neutral(self):
        p = S2Paper.from_api({"paperId": "x", "publicationTypes": ["Somethingnew"]})
        assert p.evidence_weight == 1.0

    def test_paper_dict_matches_codebase_shape(self):
        d = S2Paper.from_api(RAW_PAPER).to_paper_dict()
        for key in ("title", "summary", "authors", "url", "published", "source"):
            assert key in d
        assert d["source"] == "semanticscholar"
        assert d["citation_count"] == 412


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------

class TestClient:
    @pytest.mark.asyncio
    async def test_search_returns_parsed_papers(self):
        client = _client_returning({"total": 1, "data": [RAW_PAPER]})
        papers = await client.search("KRAS G12C")
        assert len(papers) == 1 and papers[0].title.startswith("KRAS")

    @pytest.mark.asyncio
    async def test_search_encodes_filters(self):
        calls: list = []
        client = _client_returning({"data": []}, calls)
        await client.search("x", year="2020-", min_citation_count=10,
                            publication_types=["Review"], open_access_only=True)
        url = calls[0][0]
        assert "year=2020-" in url
        assert "minCitationCount=10" in url
        assert "publicationTypes=Review" in url
        assert "openAccessPdf" in url

    @pytest.mark.asyncio
    async def test_empty_response_yields_empty_list(self):
        assert await _client_returning({"total": 0, "data": []}).search("x") == []

    @pytest.mark.asyncio
    async def test_malformed_response_does_not_raise(self):
        assert await _client_returning("not a dict").search("x") == []

    @pytest.mark.asyncio
    async def test_bulk_search_follows_pagination_tokens(self):
        pages = [
            {"data": [RAW_PAPER] * 3, "token": "tok1"},
            {"data": [RAW_PAPER] * 2, "token": None},
        ]
        state = {"i": 0}

        def payload(url):
            page = pages[min(state["i"], len(pages) - 1)]
            state["i"] += 1
            return page

        papers = await _client_returning(payload).search_bulk("x", limit=10)
        assert len(papers) == 5
        assert state["i"] == 2

    @pytest.mark.asyncio
    async def test_bulk_search_respects_the_limit(self):
        papers = await _client_returning(
            {"data": [RAW_PAPER] * 50, "token": None}
        ).search_bulk("x", limit=7)
        assert len(papers) == 7

    @pytest.mark.asyncio
    async def test_batch_chunks_at_500(self):
        calls: list = []
        client = _client_returning(lambda url: [RAW_PAPER] * 500, calls)
        await client.get_papers_batch([f"DOI:10.1000/{i}" for i in range(1200)])
        assert len(calls) == 3   # 500 + 500 + 200
        assert all(method == "POST" for _, method in calls)

    @pytest.mark.asyncio
    async def test_batch_preserves_positions_for_misses(self):
        client = _client_returning([RAW_PAPER, None, RAW_PAPER])
        got = await client.get_papers_batch(["a", "b", "c"])
        assert got[1] is None
        assert got[0] is not None and got[2] is not None

    @pytest.mark.asyncio
    async def test_references_unwrap_citedPaper(self):
        client = _client_returning({"data": [{"citedPaper": RAW_PAPER}]})
        refs = await client.get_references("abc123")
        assert len(refs) == 1 and refs[0].paper_id == "abc123"

    @pytest.mark.asyncio
    async def test_citations_unwrap_citingPaper(self):
        client = _client_returning({"data": [{"citingPaper": RAW_PAPER}]})
        assert len(await client.get_citations("abc123")) == 1

    @pytest.mark.asyncio
    async def test_recommendations_unwrap_correctly(self):
        client = _client_returning({"recommendedPapers": [RAW_PAPER, RAW_PAPER]})
        assert len(await client.recommend("abc123")) == 2

    @pytest.mark.asyncio
    async def test_id_prefixes_are_not_over_escaped(self):
        """DOI:10.1038/x must keep its colon; the slash must be escaped."""
        calls: list = []
        await _client_returning(RAW_PAPER, calls).get_paper("DOI:10.1038/s41586-024-1")
        url = calls[0][0]
        assert "DOI:" in url
        assert "10.1038%2F" in url

    def test_unauthenticated_clients_rate_limit_harder(self):
        import os

        saved = {k: os.environ.pop(k, None)
                 for k in ("S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY")}
        try:
            anon = SemanticScholarClient()
            keyed = SemanticScholarClient(api_key="k")
            assert anon._limiter.min_interval > keyed._limiter.min_interval
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_api_key_is_sent_as_header(self):
        assert SemanticScholarClient(api_key="secret")._headers()["x-api-key"] == "secret"

    def test_no_key_means_no_header(self):
        import os

        saved = {k: os.environ.pop(k, None)
                 for k in ("S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY")}
        try:
            assert "x-api-key" not in SemanticScholarClient()._headers()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    @pytest.mark.asyncio
    async def test_rate_limiter_actually_delays(self):
        from utils.semantic_scholar import _RateLimiter

        limiter = _RateLimiter(rate_per_second=50.0)
        start = asyncio.get_event_loop().time()
        for _ in range(4):
            await limiter.acquire()
        assert asyncio.get_event_loop().time() - start >= 0.05


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

class TestEnrichment:
    @pytest.mark.asyncio
    async def test_fills_the_citation_count_nothing_populated(self):
        """quality_weight read citation_count; arXiv/PubMed never supply it."""
        papers = [{"title": "X", "doi": "10.1038/s41586-024-0001", "source": "pubmed"}]
        assert "citation_count" not in papers[0]

        enriched = await _client_returning([RAW_PAPER]).enrich_papers(papers)
        assert enriched[0]["citation_count"] == 412
        assert enriched[0]["evidence_weight"] == PUBLICATION_TYPE_WEIGHT["MetaAnalysis"]

    @pytest.mark.asyncio
    async def test_resolves_arxiv_and_pmid_identifiers(self):
        calls: list = []
        client = _client_returning([RAW_PAPER, RAW_PAPER], calls)
        await client.enrich_papers([
            {"title": "A", "url": "https://arxiv.org/abs/2401.01234"},
            {"title": "B", "url": "https://pubmed.ncbi.nlm.nih.gov/38123456/"},
        ])
        assert len(calls) == 1, "identifiers must resolve in one batched call"

    @pytest.mark.asyncio
    async def test_papers_without_identifiers_pass_through(self):
        papers = [{"title": "No identifier anywhere"}]
        assert await _client_returning([]).enrich_papers(papers) == papers

    @pytest.mark.asyncio
    async def test_enrichment_never_drops_papers(self):
        papers = [
            {"title": "A", "doi": "10.1038/s41586-024-0001"},
            {"title": "B"},
        ]
        assert len(await _client_returning([None]).enrich_papers(papers)) == 2

    @pytest.mark.asyncio
    async def test_api_failure_leaves_the_corpus_intact(self):
        client = SemanticScholarClient(api_key="k")

        async def boom(*a, **k):
            raise RateLimited("quota")

        client.get_papers_batch = boom  # type: ignore[assignment]
        papers = [{"title": "A", "doi": "10.1038/s41586-024-0001"}]
        assert await client.enrich_papers(papers) == papers

    def test_evidence_hierarchy_reaches_quality_weight(self):
        meta = {"published": "2024", "source": "pubmed",
                "influential_citation_count": 38, "evidence_weight": 1.4}
        case = {"published": "2024", "source": "pubmed",
                "citation_count": 2, "evidence_weight": 0.7}
        assert quality_weight(meta) > quality_weight(case) * 2


# ---------------------------------------------------------------------------
# Grounded novelty
# ---------------------------------------------------------------------------

def _hyp(title="Compound X inhibits KRAS G12C in MOLM-13", mechanism="", method="llm-generated"):
    h = Hypothesis(title=title, description="d", mechanism=mechanism or "covalent binding")
    h.generation_method = method
    h.testable_predictions = ["IC50 below 5 uM in MOLM-13"]
    return h


class TestNoveltyIsGrounded:
    def test_query_uses_substance_not_framing(self):
        q = novelty.build_prior_art_query(_hyp())
        assert "KRAS" in q
        assert "novel" not in q.lower()

    @pytest.mark.asyncio
    async def test_generation_method_no_longer_determines_novelty(self):
        """The regression: 0.75 for 'llm', 0.55 for 'simulated'."""
        client = _client_returning({"data": []})
        scores = {}
        for method in ("llm-generated", "simulated", "evolved", "combined"):
            report = await novelty.assess_novelty(_hyp(method=method), s2_client=client)
            scores[method] = report.score
        assert len(set(scores.values())) == 1, (
            f"novelty still varies with generation_method: {scores}"
        )

    @pytest.mark.asyncio
    async def test_near_identical_prior_art_collapses_the_score(self):
        twin = dict(RAW_PAPER, title="Compound X inhibits KRAS G12C in MOLM-13",
                    abstract="Covalent binding of compound X to KRAS G12C in MOLM-13.")
        report = await novelty.assess_novelty(
            _hyp(), s2_client=_client_returning({"data": [twin]}),
        )
        assert report.searched
        assert report.score < 0.4
        assert report.level in ("low", "medium")

    @pytest.mark.asyncio
    async def test_unrelated_results_leave_novelty_high(self):
        unrelated = dict(RAW_PAPER, title="Gut microbiome composition in depression",
                         abstract="Faecal samples from cohort studies.")
        report = await novelty.assess_novelty(
            _hyp(), s2_client=_client_returning({"data": [unrelated]}),
        )
        assert report.score > 0.6

    @pytest.mark.asyncio
    async def test_failed_search_reports_unknown_not_a_default(self):
        """The distinction the old code could not express."""
        client = SemanticScholarClient(api_key="k")

        async def boom(*a, **k):
            raise RateLimited("quota exhausted")

        client.search = boom  # type: ignore[assignment]
        report = await novelty.assess_novelty(_hyp(), s2_client=client)
        assert report.searched is False
        assert report.level == "unknown"
        assert "NOT ASSESSED" in report.render()

    @pytest.mark.asyncio
    async def test_report_is_auditable(self):
        """A scalar is useless to a reviewer; the nearest papers are not."""
        report = await novelty.assess_novelty(
            _hyp(), s2_client=_client_returning({"data": [RAW_PAPER]}),
        )
        rendered = report.render()
        assert RAW_PAPER["title"][:30] in rendered
        assert RAW_PAPER["url"] in rendered

    @pytest.mark.asyncio
    async def test_apply_report_flags_probable_prior_art(self):
        twin = dict(RAW_PAPER, title="Compound X inhibits KRAS G12C in MOLM-13",
                    abstract="Compound X inhibits KRAS G12C in MOLM-13 cells.")
        h = _hyp()
        report = await novelty.assess_novelty(h, s2_client=_client_returning({"data": [twin]}))
        novelty.apply_report(h, report)
        assert h.novelty_level in ("low", "medium")
        assert h.novelty_report["searched"] is True
        assert any("prior art" in lim.lower() for lim in h.limitations)

    @pytest.mark.asyncio
    async def test_known_graph_edge_reduces_novelty(self):
        class Graph:
            @staticmethod
            def has_edge(s, p, o):
                return True

        h = _hyp()
        h.subject, h.predicate, h.object = "CompoundX", "inhibits", "KRAS"
        with_graph = await novelty.assess_novelty(
            h, s2_client=_client_returning({"data": []}), graph_agent=Graph(),
        )
        without = await novelty.assess_novelty(
            _hyp(), s2_client=_client_returning({"data": []}),
        )
        assert with_graph.score < without.score

    @pytest.mark.asyncio
    async def test_batch_assessment_bounds_concurrency(self):
        in_flight = {"now": 0, "peak": 0}
        client = SemanticScholarClient(api_key="k")

        async def tracked_search(*a, **k):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            await asyncio.sleep(0.01)
            in_flight["now"] -= 1
            return []

        client.search = tracked_search  # type: ignore[assignment]
        reports = await novelty.assess_many(
            [_hyp(f"H{i}") for i in range(9)], s2_client=client, max_concurrent=3,
        )
        assert len(reports) == 9
        assert in_flight["peak"] <= 3
