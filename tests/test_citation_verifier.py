"""
tests/test_citation_verifier.py
Offline tests for the citation verifier — all network calls are stubbed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from utils import citation_verifier as cv
from utils.citation_verifier import (
    CitationResult,
    apply_verification_penalty,
    extract_citation_ids,
    verification_score,
    verify_arxiv_id,
    verify_doi,
    verify_pmid,
    verify_text,
)


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

class TestExtract:
    def test_arxiv_canonical(self):
        ids = extract_citation_ids("See arXiv:2103.12345 for details.")
        assert ids["arxiv"] == ["2103.12345"]

    def test_arxiv_versioned(self):
        ids = extract_citation_ids("Smith et al. (arXiv: 2403.00001v2).")
        assert ids["arxiv"] == ["2403.00001v2"]

    def test_arxiv_url_form(self):
        ids = extract_citation_ids("https://arxiv.org/abs/1801.00001")
        assert "1801.00001" in ids["arxiv"]

    def test_arxiv_pdf_form(self):
        ids = extract_citation_ids("https://arxiv.org/pdf/1801.00001v3.pdf")
        assert "1801.00001v3" in ids["arxiv"]

    def test_doi_basic(self):
        ids = extract_citation_ids("see 10.1038/nature12373 for the paper")
        assert ids["doi"] == ["10.1038/nature12373"]

    def test_doi_strips_trailing_punct(self):
        ids = extract_citation_ids("...as shown in 10.1038/nature12373.")
        assert ids["doi"] == ["10.1038/nature12373"]

    def test_pmid_explicit(self):
        ids = extract_citation_ids("PMID: 38218645")
        assert ids["pmid"] == ["38218645"]

    def test_pmid_url(self):
        ids = extract_citation_ids("https://pubmed.ncbi.nlm.nih.gov/12345678/")
        assert ids["pmid"] == ["12345678"]

    def test_no_citations(self):
        ids = extract_citation_ids("This text has no citations.")
        assert ids == {"arxiv": [], "doi": [], "pmid": []}

    def test_empty_text(self):
        ids = extract_citation_ids("")
        assert ids == {"arxiv": [], "doi": [], "pmid": []}

    def test_mixed_schemes(self):
        text = (
            "Foo arXiv:2103.12345, see also 10.1038/nature12373 "
            "and PMID: 38218645."
        )
        ids = extract_citation_ids(text)
        assert ids["arxiv"] == ["2103.12345"]
        assert ids["doi"] == ["10.1038/nature12373"]
        assert ids["pmid"] == ["38218645"]

    def test_dedup(self):
        text = "arXiv:2103.12345 again arXiv:2103.12345"
        ids = extract_citation_ids(text)
        assert ids["arxiv"] == ["2103.12345"]

    def test_bare_id_only_when_arxiv_word_present(self):
        # Bare 4-digit.5-digit ID without "arxiv" nearby must NOT be picked up
        ids = extract_citation_ids("Random number 2103.12345 in text")
        assert ids["arxiv"] == []


# ---------------------------------------------------------------------------
# Per-scheme verifiers (network mocked)
# ---------------------------------------------------------------------------

class TestPerSchemeVerify:
    def test_arxiv_verify_success(self):
        with patch.object(cv, "_resolve", new=AsyncMock(return_value=200)):
            r = asyncio.run(verify_arxiv_id("2103.12345"))
        assert isinstance(r, CitationResult)
        assert r.verified is True
        assert r.type == "arxiv"
        assert "arxiv.org/abs/2103.12345" in r.source_url

    def test_arxiv_verify_failure(self):
        with patch.object(cv, "_resolve", new=AsyncMock(return_value=404)):
            r = asyncio.run(verify_arxiv_id("9999.99999"))
        assert r.verified is False
        assert "404" in r.error

    def test_doi_verify_success(self):
        with patch.object(cv, "_resolve", new=AsyncMock(return_value=302)):
            r = asyncio.run(verify_doi("10.1038/nature12373"))
        assert r.verified is True

    def test_pmid_verify_network_error(self):
        with patch.object(cv, "_resolve", new=AsyncMock(return_value=0)):
            r = asyncio.run(verify_pmid("12345"))
        assert r.verified is False


# ---------------------------------------------------------------------------
# Aggregate text verification & scoring
# ---------------------------------------------------------------------------

class TestVerifyText:
    def test_no_citations_returns_empty(self):
        result = asyncio.run(verify_text("plain text"))
        assert result == []

    def test_partial_verification(self):
        async def fake_resolve(url, timeout=5.0):
            # Real arXiv ID resolves; fake DOI does not.
            if "arxiv" in url:
                return 200
            return 404

        with patch.object(cv, "_resolve", new=fake_resolve):
            results = asyncio.run(verify_text(
                "real arXiv:2103.12345 and fake 10.9999/notreal."
            ))

        assert len(results) == 2
        verified = [r for r in results if r.verified]
        unverified = [r for r in results if not r.verified]
        assert len(verified) == 1 and verified[0].type == "arxiv"
        assert len(unverified) == 1 and unverified[0].type == "doi"


class TestScoring:
    def test_score_no_citations_is_one(self):
        assert verification_score([]) == 1.0

    def test_score_all_verified(self):
        results = [CitationResult("a", "arxiv", True), CitationResult("b", "doi", True)]
        assert verification_score(results) == 1.0

    def test_score_half_verified(self):
        results = [CitationResult("a", "arxiv", True), CitationResult("b", "doi", False)]
        assert verification_score(results) == 0.5

    def test_score_all_hallucinated(self):
        results = [CitationResult("a", "arxiv", False), CitationResult("b", "doi", False)]
        assert verification_score(results) == 0.0

    def test_penalty_no_citations_keeps_elo(self):
        assert apply_verification_penalty(1500.0, []) == 1500.0

    def test_penalty_all_hallucinated_max(self):
        bad = [CitationResult("a", "arxiv", False)]
        assert apply_verification_penalty(1500.0, bad, max_penalty=200) == 1300.0

    def test_penalty_partial(self):
        results = [
            CitationResult("a", "arxiv", True),
            CitationResult("b", "doi", False),
        ]
        # 50 % verified ⇒ half penalty
        assert apply_verification_penalty(1500.0, results, max_penalty=200) == 1400.0
