"""
Tests for the closed empirical loop: pre-registration → measurement → verdict.

These are deliberately written as *regression* tests against the specific
defects that were found, so a future refactor that reintroduces them fails
loudly rather than silently.
"""

from __future__ import annotations

import pytest

from agents.preregistration import PreregistrationAgent
from models.experiment import ExperimentKind, ExperimentRun, Measurement, VerdictStatus
from models.hypothesis import Hypothesis, Prediction
from utils.adjudication import (
    RESULTS_MARKER,
    adjudicate,
    build_prediction_contract,
    convert,
    format_verdict_report,
    match_measurement,
    parse_measurements,
    units_compatible,
)


def _hyp(**preds) -> Hypothesis:
    """Build a hypothesis with one registered prediction per kwarg."""
    h = Hypothesis(title="Test hypothesis")
    h.falsifiable_predictions = [
        Prediction(quantity=q, **cfg) for q, cfg in preds.items()
    ]
    return h


# ---------------------------------------------------------------------------
# The bug that motivated this module
# ---------------------------------------------------------------------------

class TestGrepRegression:
    """The old logic decided refutation with substring matching on stdout."""

    OLD_TRIGGERS = ("fail", "not support", "refute", "reject")

    @staticmethod
    def _old_grep_verdict(text: str) -> bool:
        low = text.lower()
        return any(t in low for t in TestGrepRegression.OLD_TRIGGERS)

    def test_failed_to_reject_is_not_a_refutation(self):
        """'failed to reject the null' means the OPPOSITE of refuted."""
        stdout = (
            "Welch t-test: t=0.41, p=0.68. We failed to reject the null hypothesis.\n"
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "IC50", '
            '"observed": 2.1, "unit": "uM", "test": "welch_t", "p_value": 0.68}]}'
        )
        # The old logic fires on this string — twice over ("fail", "reject").
        assert self._old_grep_verdict(stdout) is True

        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.3, "unit": "uM",
                       "refuting_threshold": 1.0})
        verdicts = adjudicate(h, parse_measurements(stdout),
                              kind=ExperimentKind.REAL_DATA_ANALYSIS)
        assert len(verdicts) == 1
        assert verdicts[0].status is VerdictStatus.CORROBORATED

    def test_quantitative_refutation_without_trigger_words(self):
        """A clear refutation the old grep would have missed entirely."""
        stdout = (
            "Measured half-maximal inhibitory concentration: 47.3 uM.\n"
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "IC50", '
            '"observed": 47.3, "unit": "uM"}]}'
        )
        assert self._old_grep_verdict(stdout) is False  # old logic: silent

        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.3, "unit": "uM",
                       "refuting_threshold": 1.0})
        verdicts = adjudicate(h, parse_measurements(stdout),
                              kind=ExperimentKind.REAL_DATA_ANALYSIS)
        assert verdicts[0].status is VerdictStatus.REFUTED
        assert verdicts[0].deviation == pytest.approx(45.3)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_no_marker_yields_nothing(self):
        assert parse_measurements("just some prose, no marker here") == []

    def test_last_marker_wins(self):
        """A script echoing the contract in a docstring must not confuse us."""
        text = (
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "X", "observed": 1.0}}]}}\n'
            "...actual run...\n"
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "X", "observed": 9.0}}]}}'
        )
        got = parse_measurements(text)
        assert len(got) == 1 and got[0].observed == 9.0

    def test_malformed_payload_degrades_to_empty(self):
        assert parse_measurements(f"{RESULTS_MARKER} not json at all") == []

    def test_measurement_without_numeric_observed_is_skipped(self):
        text = (
            f'{RESULTS_MARKER} {{"measurements": ['
            '{"quantity": "A", "observed": "n/a"}, '
            '{"quantity": "B", "observed": 3.0, "unit": "uM"}]}'
        )
        got = parse_measurements(text)
        assert [m.quantity for m in got] == ["B"]

    def test_full_field_extraction(self):
        text = (
            f'{RESULTS_MARKER} {{"measurements": [{{"quantity": "viability", '
            '"observed": 0.42, "unit": "ratio", "ci_low": 0.35, "ci_high": 0.49, '
            '"n": 96, "test": "mann_whitney", "p_value": 0.003}]}'
        )
        m = parse_measurements(text)[0]
        assert (m.n, m.test, m.p_value) == (96, "mann_whitney", 0.003)
        assert (m.ci_low, m.ci_high) == (0.35, 0.49)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

class TestUnits:
    def test_same_family(self):
        assert units_compatible("uM", "nM")
        assert units_compatible("mg", "kg")

    def test_different_family(self):
        assert not units_compatible("uM", "hours")

    def test_conversion(self):
        assert convert(1000.0, "nM", "uM") == pytest.approx(1.0)
        assert convert(1.0, "uM", "nM") == pytest.approx(1000.0)

    def test_unknown_units_need_exact_match(self):
        assert units_compatible("cells/mL", "cells/mL")
        assert not units_compatible("cells/mL", "colonies")

    def test_unit_mismatch_yields_invalid_not_a_silent_comparison(self):
        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 1.0})
        # 2000 nM == 2 uM, but reported as "hours" — a real mismatch.
        v = adjudicate(h, [Measurement(quantity="IC50", observed=2000, unit="hours")],
                       kind=ExperimentKind.REAL_DATA_ANALYSIS)[0]
        assert v.status is VerdictStatus.INVALID

    def test_conversion_applied_before_comparison(self):
        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 1.0})
        v = adjudicate(h, [Measurement(quantity="IC50", observed=2100, unit="nM")],
                       kind=ExperimentKind.REAL_DATA_ANALYSIS)[0]
        assert v.status is VerdictStatus.CORROBORATED
        assert v.observed_value == pytest.approx(2.1)


# ---------------------------------------------------------------------------
# The simulation guard
# ---------------------------------------------------------------------------

class TestSimulationCannotCorroborate:
    """A simulation agreeing with its own hypothesis is not evidence."""

    def test_simulation_agreement_earns_no_credit(self):
        h = _hyp(effect={"expected_value": 0.5, "ci": 0.05, "unit": "",
                         "refuting_threshold": 0.2})
        v = adjudicate(h, [Measurement(quantity="effect", observed=0.52)],
                       kind=ExperimentKind.DRY_RUN_SIMULATION)[0]
        assert v.status is VerdictStatus.CONSISTENT_UNSCORED
        assert v.status is not VerdictStatus.CORROBORATED

    def test_simulation_can_still_refute(self):
        h = _hyp(effect={"expected_value": 0.5, "ci": 0.05, "unit": "",
                         "refuting_threshold": 0.2})
        v = adjudicate(h, [Measurement(quantity="effect", observed=-3.0)],
                       kind=ExperimentKind.DRY_RUN_SIMULATION)[0]
        assert v.status is VerdictStatus.REFUTED

    def test_real_data_can_corroborate(self):
        h = _hyp(effect={"expected_value": 0.5, "ci": 0.05, "unit": "",
                         "refuting_threshold": 0.2})
        v = adjudicate(h, [Measurement(quantity="effect", observed=0.52)],
                       kind=ExperimentKind.REAL_DATA_ANALYSIS)[0]
        assert v.status is VerdictStatus.CORROBORATED

    def test_kind_capability_flags(self):
        assert not ExperimentKind.DRY_RUN_SIMULATION.can_corroborate
        assert ExperimentKind.DRY_RUN_SIMULATION.can_refute
        assert ExperimentKind.REAL_DATA_ANALYSIS.can_corroborate
        assert not ExperimentKind.INFEASIBLE.can_refute


# ---------------------------------------------------------------------------
# Untested predictions must stay visible
# ---------------------------------------------------------------------------

class TestUntestedIsNotSupport:
    def test_missing_measurement_is_untested(self):
        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 1.0})
        v = adjudicate(h, [], kind=ExperimentKind.REAL_DATA_ANALYSIS)[0]
        assert v.status is VerdictStatus.UNTESTED

    def test_every_prediction_gets_a_verdict(self):
        h = _hyp(
            A={"expected_value": 1.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
            B={"expected_value": 2.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
            C={"expected_value": 3.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
        )
        verdicts = adjudicate(h, [Measurement(quantity="B", observed=2.0, unit="uM")],
                              kind=ExperimentKind.REAL_DATA_ANALYSIS)
        assert len(verdicts) == 3
        assert sum(1 for v in verdicts if v.status is VerdictStatus.UNTESTED) == 2

    def test_untested_contributes_zero_evidence(self):
        run = ExperimentRun(kind=ExperimentKind.REAL_DATA_ANALYSIS)
        h = _hyp(A={"expected_value": 1.0, "ci": 0.1, "unit": "uM",
                    "refuting_threshold": 0.5})
        run.verdicts = adjudicate(h, [], kind=ExperimentKind.REAL_DATA_ANALYSIS)
        assert run.evidential_weight == 0.0
        assert run.refuted is False

    def test_unfalsifiable_prediction_flagged(self):
        h = _hyp(vague={"expected_value": 0.0, "ci": 0.0, "unit": "arbitrary",
                        "refuting_threshold": 0.0})
        v = adjudicate(h, [Measurement(quantity="vague", observed=5.0, unit="arbitrary")],
                       kind=ExperimentKind.REAL_DATA_ANALYSIS)[0]
        assert v.status is VerdictStatus.UNFALSIFIABLE


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class TestMatching:
    def test_fuzzy_match_on_formatting_differences(self):
        pred = Prediction(quantity="IC_50 (MOLM-13)", expected_value=1.0,
                          unit="uM", refuting_threshold=0.5)
        m, idx = match_measurement(pred, [Measurement(quantity="ic50 molm13", observed=1.0)])
        assert m is not None and idx == 0

    def test_one_measurement_cannot_serve_two_predictions(self):
        h = _hyp(
            IC50={"expected_value": 1.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
            IC50b={"expected_value": 1.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
        )
        verdicts = adjudicate(h, [Measurement(quantity="IC50", observed=1.0, unit="uM")],
                              kind=ExperimentKind.REAL_DATA_ANALYSIS)
        scored = [v for v in verdicts if v.status is VerdictStatus.CORROBORATED]
        assert len(scored) == 1

    def test_unrelated_quantity_does_not_match(self):
        pred = Prediction(quantity="IC50", expected_value=1.0, unit="uM",
                          refuting_threshold=0.5)
        m, _ = match_measurement(pred, [Measurement(quantity="body_temperature", observed=37.0)])
        assert m is None


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

class TestReporting:
    def test_report_lists_every_status(self):
        h = _hyp(
            A={"expected_value": 1.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
            B={"expected_value": 2.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
        )
        verdicts = adjudicate(h, [Measurement(quantity="A", observed=9.0, unit="uM")],
                              kind=ExperimentKind.REAL_DATA_ANALYSIS)
        report = format_verdict_report(verdicts)
        assert "refuted" in report and "untested" in report

    def test_contract_lists_registered_quantities(self):
        h = _hyp(IC50={"expected_value": 2.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 1.0})
        contract = build_prediction_contract(h)
        assert 'quantity="IC50"' in contract and "uM" in contract

    def test_run_summary_flags_uncredited_simulation(self):
        h = _hyp(A={"expected_value": 1.0, "ci": 0.1, "unit": "uM",
                    "refuting_threshold": 0.5})
        run = ExperimentRun(kind=ExperimentKind.DRY_RUN_SIMULATION)
        run.verdicts = adjudicate(h, [Measurement(quantity="A", observed=1.0, unit="uM")],
                                  kind=ExperimentKind.DRY_RUN_SIMULATION)
        assert "corroboration not credited" in run.summary()


# ---------------------------------------------------------------------------
# Pre-registration integrity (the always-False bug)
# ---------------------------------------------------------------------------

class TestPreregistrationIntegrity:
    def test_hash_is_stable_across_calls(self):
        """Hashing datetime.now() made verify_integrity() always return False."""
        preds = [Prediction(quantity="IC50", expected_value=1.0, ci=0.1,
                            unit="uM", refuting_threshold=0.5)]
        h1 = PreregistrationAgent._compute_prediction_hash(preds)
        h2 = PreregistrationAgent._compute_prediction_hash(preds)
        assert h1 == h2

    def test_intact_bundle_verifies(self):
        h = _hyp(IC50={"expected_value": 1.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 0.5})
        h.prediction_hash = PreregistrationAgent._compute_prediction_hash(
            h.falsifiable_predictions
        )
        ok, detail = PreregistrationAgent.check_integrity(h)
        assert ok, detail

    def test_harking_is_detected(self):
        h = _hyp(IC50={"expected_value": 1.0, "ci": 0.1, "unit": "uM",
                       "refuting_threshold": 0.5})
        h.prediction_hash = PreregistrationAgent._compute_prediction_hash(
            h.falsifiable_predictions
        )
        h.falsifiable_predictions[0].expected_value = 47.0   # moved post hoc
        ok, detail = PreregistrationAgent.check_integrity(h)
        assert not ok and "TAMPERED" in detail

    def test_reordering_is_not_tampering(self):
        h = _hyp(
            A={"expected_value": 1.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
            B={"expected_value": 2.0, "ci": 0.1, "unit": "uM", "refuting_threshold": 0.5},
        )
        h.prediction_hash = PreregistrationAgent._compute_prediction_hash(
            h.falsifiable_predictions
        )
        h.falsifiable_predictions.reverse()
        ok, _ = PreregistrationAgent.check_integrity(h)
        assert ok

    def test_never_registered_is_distinguishable_from_tampered(self):
        ok, detail = PreregistrationAgent.check_integrity(Hypothesis(title="x"))
        assert not ok
        assert "never" in detail
        assert "TAMPERED" not in detail   # the two cases must stay distinguishable
