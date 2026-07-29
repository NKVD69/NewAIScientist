"""
utils/adjudication.py — Confront pre-registered predictions with observed measurements.

This module replaces the substring search that used to decide whether a
hypothesis had been refuted::

    # OLD — co_scientist.py
    if "fail" in results.lower() or "reject" in results.lower():
        refuted = True

That test fires on ``"failed to reject the null hypothesis"`` — the standard
phrase for *absence of evidence against* — and misses an unambiguous
quantitative refutation such as ``"observed IC50 = 47 uM vs predicted 2 uM"``.

The replacement is a per-prediction confrontation using the machinery that
already existed but was never called: ``Prediction.is_refuted_by()``.

Three guards make the verdicts trustworthy:

1. **Unit compatibility.** A measurement in ``nM`` is converted to the
   prediction's ``uM`` before comparison; incompatible dimensions yield
   ``INVALID`` rather than a silent apples-to-oranges comparison.
2. **Experiment kind.** A ``DRY_RUN_SIMULATION`` may refute but never
   corroborate (see ``models.experiment``).
3. **Explicit "untested".** A prediction with no matching measurement is
   recorded as ``UNTESTED`` and contributes zero evidence — it is *not*
   silently treated as support, which is what the old code did.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher

from models.experiment import (
    ExperimentKind,
    Measurement,
    Verdict,
    VerdictStatus,
)
from models.hypothesis import Hypothesis, Prediction

logger = logging.getLogger(__name__)

#: Marker the generated experiment script must print on its last line.
RESULTS_MARKER = "RESULTS_JSON:"

#: Instruction block appended to every experiment-generation prompt.
RESULTS_CONTRACT = f"""
Your script MUST end by printing exactly one line with this shape:

{RESULTS_MARKER} {{"measurements": [
    {{"quantity": "<name matching a registered prediction>",
     "observed": <float>, "unit": "<unit>",
     "ci_low": <float|null>, "ci_high": <float|null>,
     "n": <int|null>, "test": "<test name>", "p_value": <float|null>}}
]}}

Rules for this line:
- It must be valid JSON on a SINGLE line, printed last, nothing after it.
- "quantity" must reuse the exact quantity names given to you below.
- Report the measured value in "observed" even when the result is negative
  or contradicts the hypothesis. Do NOT omit unfavourable measurements.
- If a quantity could not be measured, omit it entirely rather than
  inventing a value.
"""


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------

#: Conversion factors to a canonical base unit, grouped by dimension.
_UNIT_FAMILIES: dict[str, dict[str, float]] = {
    "molar": {
        "m": 1.0, "mol/l": 1.0, "molar": 1.0,
        "mm": 1e-3, "mmol/l": 1e-3,
        "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "umol/l": 1e-6, "micromolar": 1e-6,
        "nm": 1e-9, "nmol/l": 1e-9, "nanomolar": 1e-9,
        "pm": 1e-12, "pmol/l": 1e-12,
    },
    "mass": {
        "kg": 1e3, "g": 1.0, "mg": 1e-3, "ug": 1e-6, "µg": 1e-6, "ng": 1e-9,
        "da": 1.0, "dalton": 1.0, "g/mol": 1.0,
    },
    "time": {
        "s": 1.0, "sec": 1.0, "second": 1.0,
        "min": 60.0, "minute": 60.0,
        "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
        "day": 86400.0, "d": 86400.0,
    },
    "dimensionless": {
        "": 1.0, "-": 1.0, "ratio": 1.0, "fold": 1.0, "x": 1.0,
        "au": 1.0, "a.u.": 1.0, "count": 1.0, "n": 1.0, "score": 1.0,
        "log2fc": 1.0, "logfc": 1.0, "z": 1.0, "cohen_d": 1.0, "d": 1.0,
    },
    "percent": {"%": 1.0, "percent": 1.0, "pct": 1.0},
}

#: ``nm`` is ambiguous (nanomolar vs nanometre). Molar wins because the
#: biomedical prediction vocabulary is dominated by concentrations; a
#: length prediction should spell out ``nanometre``.
_LENGTH = {"nanometre": 1e-9, "nanometer": 1e-9, "um_length": 1e-6,
           "micrometre": 1e-6, "mm_length": 1e-3, "cm": 1e-2, "m_length": 1.0,
           "angstrom": 1e-10, "a": 1e-10}
_UNIT_FAMILIES["length"] = _LENGTH


def _normalise_unit(unit: str) -> str:
    return (unit or "").strip().lower().replace(" ", "")


def unit_family(unit: str) -> str | None:
    """Return the dimension family of ``unit``, or None if unrecognised."""
    u = _normalise_unit(unit)
    for family, table in _UNIT_FAMILIES.items():
        if u in table:
            return family
    return None


def units_compatible(unit_a: str, unit_b: str) -> bool:
    """Whether two units describe the same physical dimension."""
    fa, fb = unit_family(unit_a), unit_family(unit_b)
    if fa is None or fb is None:
        # Unknown units: only accept an exact textual match, so that a
        # domain-specific unit ("cells/mL") still works when both sides agree.
        return _normalise_unit(unit_a) == _normalise_unit(unit_b)
    return fa == fb


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    """Convert ``value`` between two units of the same family.

    Returns None when the units are incompatible or unknown-and-different.
    """
    fa, fb = unit_family(from_unit), unit_family(to_unit)
    if fa is None or fb is None:
        if _normalise_unit(from_unit) == _normalise_unit(to_unit):
            return float(value)
        return None
    if fa != fb:
        return None
    table = _UNIT_FAMILIES[fa]
    return float(value) * table[_normalise_unit(from_unit)] / table[_normalise_unit(to_unit)]


# ---------------------------------------------------------------------------
# Parsing the RESULTS_JSON line
# ---------------------------------------------------------------------------

def _opt_float(item: dict, key: str) -> float | None:
    """Read an optional float field, tolerating nulls and junk."""
    val = item.get(key)
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def _opt_int(item: dict, key: str) -> int | None:
    """Read an optional int field, tolerating nulls and junk."""
    val = item.get(key)
    try:
        return None if val is None else int(val)
    except (TypeError, ValueError):
        return None


def parse_measurements(stdout: str) -> list[Measurement]:
    """Extract ``Measurement`` objects from an experiment script's stdout.

    Scans for the ``RESULTS_JSON:`` marker (last occurrence wins, so a script
    that echoes the contract in a docstring does not confuse us). Returns an
    empty list when the marker is absent or unparseable — the caller then
    records every prediction as ``UNTESTED``, which is the correct, honest
    outcome rather than a guess.
    """
    if not stdout or RESULTS_MARKER not in stdout:
        return []

    tail = stdout.rsplit(RESULTS_MARKER, 1)[1].strip()

    payload = None
    # The JSON should be the remainder of that line, but be forgiving about
    # trailing prose and about the object spilling onto following lines.
    for candidate in (tail.splitlines()[0] if tail else "", tail):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                    break
                except json.JSONDecodeError:
                    continue

    if not isinstance(payload, dict):
        logger.warning("RESULTS_JSON marker found but payload was unparseable.")
        return []

    raw_items = payload.get("measurements", [])
    if not isinstance(raw_items, list):
        return []

    measurements: list[Measurement] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            observed = float(item["observed"])
        except (KeyError, TypeError, ValueError):
            logger.debug("Skipping measurement without numeric 'observed': %r", item)
            continue

        measurements.append(Measurement(
            quantity=str(item.get("quantity", "")).strip(),
            observed=observed,
            unit=str(item.get("unit", "")).strip(),
            ci_low=_opt_float(item, "ci_low"),
            ci_high=_opt_float(item, "ci_high"),
            n=_opt_int(item, "n"),
            test=str(item.get("test", "")).strip(),
            p_value=_opt_float(item, "p_value"),
        ))

    return measurements


# ---------------------------------------------------------------------------
# Matching predictions to measurements
# ---------------------------------------------------------------------------

_MATCH_THRESHOLD = 0.72


def _canon(text: str) -> str:
    """Loose canonical form for fuzzy quantity matching."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def match_measurement(
    prediction: Prediction,
    measurements: list[Measurement],
    used: set[int] | None = None,
) -> tuple[Measurement | None, int | None]:
    """Find the measurement corresponding to ``prediction``.

    Exact canonical match first, then fuzzy match above ``_MATCH_THRESHOLD``.
    ``used`` lets the caller enforce one-measurement-per-prediction so a
    single observation cannot corroborate several predictions at once.
    """
    used = used if used is not None else set()
    target = _canon(prediction.quantity)
    if not target:
        return None, None

    for idx, m in enumerate(measurements):
        if idx in used:
            continue
        if _canon(m.quantity) == target:
            return m, idx

    best_idx, best_score = None, 0.0
    for idx, m in enumerate(measurements):
        if idx in used:
            continue
        score = SequenceMatcher(None, target, _canon(m.quantity)).ratio()
        if score > best_score:
            best_idx, best_score = idx, score

    if best_idx is not None and best_score >= _MATCH_THRESHOLD:
        return measurements[best_idx], best_idx
    return None, None


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def adjudicate(
    hypothesis: Hypothesis,
    measurements: list[Measurement],
    kind: ExperimentKind = ExperimentKind.DRY_RUN_SIMULATION,
) -> list[Verdict]:
    """Confront every pre-registered prediction with the observed measurements.

    Returns one ``Verdict`` per registered prediction — never fewer, so an
    untested prediction is always visible rather than silently dropped.
    """
    predictions: list[Prediction] = list(hypothesis.falsifiable_predictions or [])
    if not predictions:
        logger.info(
            "Hypothesis '%s' has no pre-registered predictions — nothing to adjudicate.",
            (hypothesis.title or "")[:50],
        )
        return []

    verdicts: list[Verdict] = []
    used: set[int] = set()

    for pred in predictions:
        base = Verdict(
            prediction_quantity=pred.quantity,
            expected_value=pred.expected_value,
            unit=pred.unit,
            refuting_threshold=pred.refuting_threshold,
            experiment_kind=kind.value,
        )

        if kind is ExperimentKind.INFEASIBLE:
            base.status = VerdictStatus.UNTESTED
            base.reason = "experiment was not feasible; no data source reachable"
            verdicts.append(base)
            continue

        if not pred.is_falsifiable():
            base.status = VerdictStatus.UNFALSIFIABLE
            base.reason = (
                f"refuting_threshold={pred.refuting_threshold} is not usable; "
                "prediction cannot be refuted by any observation"
            )
            verdicts.append(base)
            continue

        measurement, idx = match_measurement(pred, measurements, used)
        if measurement is None:
            base.status = VerdictStatus.UNTESTED
            base.reason = f"no measurement reported for quantity '{pred.quantity}'"
            verdicts.append(base)
            continue

        used.add(idx)  # type: ignore[arg-type]

        if not units_compatible(pred.unit, measurement.unit):
            base.status = VerdictStatus.INVALID
            base.observed_value = measurement.observed
            base.reason = (
                f"unit mismatch: measured in '{measurement.unit}', "
                f"prediction registered in '{pred.unit}'"
            )
            verdicts.append(base)
            continue

        observed = convert(measurement.observed, measurement.unit, pred.unit)
        if observed is None:
            base.status = VerdictStatus.INVALID
            base.observed_value = measurement.observed
            base.reason = (
                f"could not convert {measurement.unit} → {pred.unit}"
            )
            verdicts.append(base)
            continue

        base.observed_value = observed
        base.deviation = abs(observed - pred.expected_value)

        # ↓ The call that never happened before this module existed.
        if pred.is_refuted_by(observed):
            base.status = VerdictStatus.REFUTED
            base.reason = (
                f"|{observed:.4g} - {pred.expected_value:.4g}| = "
                f"{base.deviation:.4g} {pred.unit} exceeds refuting threshold "
                f"{pred.refuting_threshold:.4g}"
            )
        elif kind.can_corroborate:
            base.status = VerdictStatus.CORROBORATED
            base.reason = (
                f"observed {observed:.4g} {pred.unit} within threshold "
                f"{pred.refuting_threshold:.4g} of expected "
                f"{pred.expected_value:.4g}"
            )
        else:
            # Simulation agreed with the prediction. This is not evidence.
            base.status = VerdictStatus.CONSISTENT_UNSCORED
            base.reason = (
                "observation is consistent with the prediction, but the run is a "
                f"'{kind.value}' and therefore cannot corroborate; "
                "no evidential credit awarded"
            )

        verdicts.append(base)

    return verdicts


def format_verdict_report(verdicts: list[Verdict]) -> str:
    """Human-readable adjudication report for logs, the UI and reviewers."""
    if not verdicts:
        return "No pre-registered predictions were available for adjudication."

    icons = {
        VerdictStatus.CORROBORATED: "✅",
        VerdictStatus.REFUTED: "❌",
        VerdictStatus.CONSISTENT_UNSCORED: "◻️",
        VerdictStatus.UNTESTED: "⬜",
        VerdictStatus.UNFALSIFIABLE: "⚠️",
        VerdictStatus.INVALID: "🚫",
    }

    lines = ["## Adjudication Report (pre-registered predictions vs measurements)", ""]
    for v in verdicts:
        icon = icons.get(v.status, "•")
        head = f"{icon} **{v.prediction_quantity or '(unnamed)'}** — {v.status.value}"
        if v.observed_value is not None and v.expected_value is not None:
            head += (
                f" · observed {v.observed_value:.4g} vs expected "
                f"{v.expected_value:.4g} {v.unit}".rstrip()
            )
        lines.append(head)
        if v.reason:
            lines.append(f"    ↳ {v.reason}")

    counts = {status: sum(1 for v in verdicts if v.status is status) for status in VerdictStatus}
    lines.append("")
    lines.append(
        "**Totals:** "
        f"{counts[VerdictStatus.CORROBORATED]} corroborated · "
        f"{counts[VerdictStatus.REFUTED]} refuted · "
        f"{counts[VerdictStatus.CONSISTENT_UNSCORED]} consistent (unscored) · "
        f"{counts[VerdictStatus.UNTESTED]} untested · "
        f"{counts[VerdictStatus.UNFALSIFIABLE]} unfalsifiable · "
        f"{counts[VerdictStatus.INVALID]} invalid"
    )
    return "\n".join(lines)


def build_prediction_contract(hypothesis: Hypothesis) -> str:
    """Render the registered quantities so the script measures the right things.

    Feeding the exact quantity names into the generation prompt is what makes
    ``match_measurement`` reliable — without it the LLM invents its own names
    and every prediction lands in ``UNTESTED``.
    """
    preds = hypothesis.falsifiable_predictions or []
    if not preds:
        return (
            "No pre-registered predictions are available for this hypothesis. "
            "Report any quantity you measure, but note that the run cannot be "
            "adjudicated."
        )

    lines = ["Pre-registered quantities you must attempt to measure:"]
    for p in preds:
        lines.append(
            f'  - quantity="{p.quantity}", unit="{p.unit}", '
            f"expected≈{p.expected_value:g}, refuted if |observed - expected| > "
            f"{p.refuting_threshold:g}"
        )
    lines.append("")
    lines.append(
        "Measure these honestly. A refutation is a valid and useful scientific "
        "outcome — do not bias the analysis toward the expected values."
    )
    return "\n".join(lines)


__all__ = [
    "RESULTS_CONTRACT",
    "RESULTS_MARKER",
    "adjudicate",
    "build_prediction_contract",
    "convert",
    "format_verdict_report",
    "match_measurement",
    "parse_measurements",
    "unit_family",
    "units_compatible",
]
