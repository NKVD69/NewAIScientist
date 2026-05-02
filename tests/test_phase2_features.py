"""
tests/test_phase2_features.py
Offline tests for the second batch of functional improvements:
  - Claim / Evidence / Prediction model
  - Falsifiability scorer
  - Swiss + information-gain pairing
  - Multi-criterion judge
  - PDF table / figure-caption extraction
  - Incremental literature refresh
  - Experiment sandbox (power analysis + entity validation)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.hypothesis import Claim, Evidence, Hypothesis, Prediction


# ===========================================================================
# #2 Claim / Evidence / Prediction model
# ===========================================================================

class TestClaimEvidenceModel:
    def test_evidence_polarity_validation(self):
        with pytest.raises(ValueError):
            Evidence(text="x", polarity=2)

    def test_evidence_confidence_validation(self):
        with pytest.raises(ValueError):
            Evidence(text="x", confidence=1.5)

    def test_claim_score_no_evidence_is_zero(self):
        c = Claim(statement="x")
        assert c.evidence_score() == 0.0

    def test_claim_score_all_supporting(self):
        c = Claim(statement="x", evidence=[
            Evidence(text="e1", polarity=1, confidence=0.8),
            Evidence(text="e2", polarity=1, confidence=0.6),
        ])
        # Both polarities are +1, weighted mean of polarity = 1.0
        assert c.evidence_score() == pytest.approx(1.0)

    def test_claim_score_mixed(self):
        c = Claim(statement="x", evidence=[
            Evidence(text="e1", polarity=1, confidence=1.0),
            Evidence(text="e2", polarity=-1, confidence=1.0),
        ])
        assert c.evidence_score() == pytest.approx(0.0)

    def test_prediction_falsifiable(self):
        p = Prediction(quantity="IC50", expected_value=10, ci=2,
                       refuting_threshold=5)
        assert p.is_falsifiable()
        assert p.is_refuted_by(20.0)
        assert not p.is_refuted_by(11.0)

    def test_prediction_unfalsifiable_when_no_threshold(self):
        p = Prediction(quantity="IC50", expected_value=10, ci=2)
        assert not p.is_falsifiable()
        assert not p.is_refuted_by(1000.0)

    def test_hypothesis_carries_claims_and_predictions(self):
        h = Hypothesis(title="t")
        assert h.claims == []
        assert h.falsifiable_predictions == []
        assert h.falsifiability_score == 0.0


# ===========================================================================
# #3 Falsifiability scorer
# ===========================================================================

class TestFalsifiability:
    def test_empty_returns_zero(self):
        from utils.falsifiability import score_hypothesis
        h = Hypothesis(title="t")
        assert score_hypothesis(h) == 0.0

    def test_text_predictions_with_numbers_only(self):
        from utils.falsifiability import score_text_predictions
        s = score_text_predictions(["The drug reduces tumor size by 30%"])
        # number + comparator + unit/% → strong signal
        assert s > 0.6

    def test_text_predictions_purely_qualitative(self):
        from utils.falsifiability import score_text_predictions
        s = score_text_predictions(["Some effect occurs"])
        assert s == 0.0

    def test_structured_prediction_dominates_free_text(self):
        from utils.falsifiability import score_hypothesis
        good = Prediction(quantity="IC50", expected_value=5, ci=1,
                          refuting_threshold=10, unit="nM",
                          rationale="Based on prior studies")
        h = Hypothesis(
            title="t",
            falsifiable_predictions=[good],
            testable_predictions=["vague claim"],
        )
        assert score_hypothesis(h) > 0.7

    def test_annotate_writes_field(self):
        from utils.falsifiability import annotate
        h = Hypothesis(title="t",
                       testable_predictions=["growth decreases by 20%"])
        annotate(h)
        assert h.falsifiability_score > 0


# ===========================================================================
# #4 Tournament pairing
# ===========================================================================

class TestSwissPairing:
    def test_basic_top_down(self):
        from utils.tournament_pairing import swiss_pairing
        pairs = swiss_pairing([("a", 1500), ("b", 1400), ("c", 1300), ("d", 1200)])
        # Top-down: 1st-2nd, 3rd-4th
        assert pairs == [("a", "b"), ("c", "d")]

    def test_avoids_rematch(self):
        from utils.tournament_pairing import swiss_pairing
        pairs = swiss_pairing(
            [("a", 1500), ("b", 1400), ("c", 1300), ("d", 1200)],
            history=[("a", "b")],
        )
        # 'a' should not be paired with 'b' again.
        assert frozenset(pairs[0]) != frozenset(("a", "b"))

    def test_odd_count_drops_one(self):
        from utils.tournament_pairing import swiss_pairing
        pairs = swiss_pairing([("a", 1500), ("b", 1400), ("c", 1300)])
        assert len(pairs) == 1

    def test_empty_input(self):
        from utils.tournament_pairing import swiss_pairing
        assert swiss_pairing([]) == []
        assert swiss_pairing([("a", 1500)]) == []


class TestInformationGainPairing:
    def test_prefers_close_elo(self):
        from utils.tournament_pairing import information_gain_pairing
        # 'a' and 'b' are close (high entropy); 'a' vs 'd' is lopsided.
        pairs = information_gain_pairing(
            [("a", 1500), ("b", 1490), ("c", 1300), ("d", 1100)],
            num_matches=1,
        )
        assert frozenset(pairs[0]) == frozenset(("a", "b"))

    def test_history_penalises_repeats(self):
        from utils.tournament_pairing import information_gain_pairing
        history = [("a", "b")] * 5
        pairs = information_gain_pairing(
            [("a", 1500), ("b", 1490), ("c", 1488)],
            num_matches=1,
            history=history,
        )
        # After heavy penalty, the 'a-b' pair should be avoided
        assert frozenset(pairs[0]) != frozenset(("a", "b"))

    def test_zero_matches_returns_empty(self):
        from utils.tournament_pairing import information_gain_pairing
        assert information_gain_pairing([("a", 1500), ("b", 1400)], 0) == []


# ===========================================================================
# #5 Multi-criterion judge
# ===========================================================================

class TestMultiCriterionJudge:
    def _agent(self):
        from agents.ranking import RankingAgent
        agent = RankingAgent(use_local_llm=False, verify_citations=False)
        agent.llm_client = None
        return agent

    def test_default_weights_normalised(self):
        agent = self._agent()
        total = sum(agent.criteria_weights.values())
        assert total == pytest.approx(1.0)

    def test_aggregate_majority_wins(self):
        agent = self._agent()
        verdicts = {
            "novelty": "A", "plausibility": "A",
            "testability": "B", "impact": "B",
        }
        # Default weights: novelty=0.25, plausibility=0.30, testability=0.25, impact=0.20
        # A score: 0.55 ; B score: 0.45 ⇒ A wins
        assert agent._aggregate_verdicts("A", "B", verdicts) == "A"

    def test_per_criterion_elo_default(self):
        agent = self._agent()
        assert agent.per_criterion_rating("nope", "novelty") == 1200.0

    def test_update_multi_elo_changes_ratings(self):
        agent = self._agent()
        a = Hypothesis(id="a", title="A")
        b = Hypothesis(id="b", title="B")
        agent._update_multi_elo(a, b, {"novelty": "a"})
        assert agent.per_criterion_rating("a", "novelty") > 1200
        assert agent.per_criterion_rating("b", "novelty") < 1200

    def test_normalise_weights_handles_zeros(self):
        from agents.ranking import RankingAgent
        norm = RankingAgent._normalise_weights({"x": 0.0, "y": 0.0})
        # All-zero defaults to keeping zeros (sum stays 0/1=0)
        assert all(v == 0.0 for v in norm.values())


# ===========================================================================
# #7 Tables / figures
# ===========================================================================

class TestFigureCaptions:
    def test_extracts_basic_caption(self):
        from rag_system import extract_figure_captions
        text = "Some prose.\nFigure 1: KRAS activates downstream MAPK signalling.\nMore text."
        caps = extract_figure_captions(text)
        assert any("KRAS" in c and "Figure 1" in c for c in caps)

    def test_extracts_multiple(self):
        from rag_system import extract_figure_captions
        text = (
            "Figure 1: First caption.\n"
            "Some body.\n"
            "Table 2: Demographic data.\n"
            "Fig. 3 — Final results.\n"
        )
        caps = extract_figure_captions(text)
        assert len(caps) >= 3

    def test_no_match(self):
        from rag_system import extract_figure_captions
        assert extract_figure_captions("Plain text without captions.") == []

    def test_empty(self):
        from rag_system import extract_figure_captions
        assert extract_figure_captions("") == []


class TestTableSerialiser:
    def test_basic(self):
        from rag_system import _serialise_table
        out = _serialise_table([["A", "B"], ["1", "2"], ["3", "4"]])
        assert "A | B" in out
        assert "1 | 2" in out

    def test_skips_empty_rows(self):
        from rag_system import _serialise_table
        out = _serialise_table([["A", "B"], [None, ""], ["1", "2"]])
        assert out.count("\n") == 1  # 2 lines


# ===========================================================================
# #8 Incremental literature refresh
# ===========================================================================

class TestLiteratureRefresh:
    def test_filter_drops_known(self):
        from utils.literature_refresh import filter_new_papers
        existing = [{"url": "u1", "title": "t1"}]
        fetched = [{"url": "u1", "title": "t1"}, {"url": "u2", "title": "t2"}]
        new = filter_new_papers(fetched, existing)
        assert len(new) == 1 and new[0]["url"] == "u2"

    def test_filter_drops_old_by_watermark(self):
        from utils.literature_refresh import filter_new_papers
        fetched = [
            {"url": "u1", "published": "2024-01-01"},
            {"url": "u2", "published": "2025-06-01"},
        ]
        new = filter_new_papers(fetched, [], last_seen="2025-01-01")
        assert len(new) == 1 and new[0]["url"] == "u2"

    def test_filter_keeps_undated(self):
        from utils.literature_refresh import filter_new_papers
        fetched = [{"url": "u1"}]  # no published date
        new = filter_new_papers(fetched, [], last_seen="2099-01-01")
        assert len(new) == 1  # err on the side of keeping

    def test_update_watermark_advances(self):
        from utils.literature_refresh import update_watermark
        last = {}
        update_watermark(last, "arxiv", [
            {"published": "2024-05-01"},
            {"published": "2025-06-15"},
        ])
        assert last["arxiv"].startswith("2025-06-15")

    def test_update_watermark_does_not_regress(self):
        from utils.literature_refresh import update_watermark
        last = {"arxiv": "2025-12-31"}
        update_watermark(last, "arxiv", [{"published": "2024-01-01"}])
        assert last["arxiv"].startswith("2025-12-31")

    def test_update_watermark_handles_missing_dates(self):
        from utils.literature_refresh import update_watermark
        last = {}
        update_watermark(last, "arxiv", [{}, {"title": "no date"}])
        assert last == {}


# ===========================================================================
# #9 Experiment sandbox
# ===========================================================================

class TestPowerAnalysis:
    def test_n_decreases_with_effect_size(self):
        from utils.experiment_sandbox import estimate_required_n
        assert estimate_required_n(0.2) > estimate_required_n(0.5)
        assert estimate_required_n(0.5) > estimate_required_n(1.0)

    def test_zero_effect_returns_minus_one(self):
        from utils.experiment_sandbox import estimate_required_n
        assert estimate_required_n(0.0) == -1

    def test_canonical_values_reasonable(self):
        # Cohen-medium effect (d=0.5), α=0.05, power=0.80 ⇒ n ≈ 64 / group
        from utils.experiment_sandbox import estimate_required_n
        n = estimate_required_n(0.5, 0.05, 0.80)
        assert 60 <= n <= 70

    def test_higher_power_requires_more_n(self):
        from utils.experiment_sandbox import estimate_required_n
        assert estimate_required_n(0.5, 0.05, 0.90) > \
               estimate_required_n(0.5, 0.05, 0.80)


class TestEntityExtraction:
    def test_uniprot_match(self):
        from utils.experiment_sandbox import extract_entities
        ids = extract_entities("Studied protein P04637 (TP53) and Q8WZ42.")
        assert "P04637" in ids["uniprot"]
        assert "Q8WZ42" in ids["uniprot"]

    def test_pubchem_cid(self):
        from utils.experiment_sandbox import extract_entities
        ids = extract_entities("Treated with CID 2244 (aspirin) and CID:5957.")
        assert "2244" in ids["pubchem"]
        assert "5957" in ids["pubchem"]

    def test_no_entities(self):
        from utils.experiment_sandbox import extract_entities
        ids = extract_entities("Just prose, no biomedical IDs here.")
        assert ids == {"uniprot": [], "pubchem": []}


class TestEntityValidation:
    def test_validate_entities_all_resolve(self):
        from utils import experiment_sandbox as es
        with patch.object(es, "_resolve", new=AsyncMock(return_value=200)):
            results = asyncio.run(es.validate_entities("Studied P04637 and CID 2244"))
        assert len(results) == 2
        assert all(r.verified for r in results)

    def test_validate_entities_mixed(self):
        from utils import experiment_sandbox as es

        async def fake_resolve(url, timeout):
            return 200 if "uniprot" in url else 404

        with patch.object(es, "_resolve", new=fake_resolve):
            results = asyncio.run(es.validate_entities("P04637 and CID 9999"))

        verified = [r for r in results if r.verified]
        assert len(verified) == 1 and verified[0].type == "uniprot"

    def test_no_entities_returns_empty(self):
        from utils import experiment_sandbox as es
        results = asyncio.run(es.validate_entities("plain text"))
        assert results == []


class TestFeasibilitySummary:
    def test_no_entities(self):
        from utils.experiment_sandbox import feasibility_summary
        out = feasibility_summary(64, [])
        assert out["required_n_per_group"] == 64
        assert out["entity_verification_rate"] == 1.0
        assert out["entities"] == []

    def test_with_entities(self):
        from utils.experiment_sandbox import EntityResult, feasibility_summary
        out = feasibility_summary(64, [
            EntityResult("P1", "uniprot", True),
            EntityResult("X", "uniprot", False),
        ])
        assert out["entity_verification_rate"] == 0.5
