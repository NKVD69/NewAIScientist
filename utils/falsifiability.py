"""
utils/falsifiability.py
Score how falsifiable a hypothesis is — Popper-style.

A hypothesis is falsifiable to the degree that:
  1. it carries quantitative predictions (`Prediction.is_falsifiable()`),
  2. those predictions cite measurable quantities (a non-empty ``unit``),
  3. and they prescribe an explicit refuting threshold.

The free-text scorer ``score_text_predictions`` is a heuristic fallback
used when no structured ``Prediction`` objects exist; it rewards
predictions that contain numbers, units, and comparators (``>``, ``<``,
``±``, ``increases``…).

The scorer never raises — a missing field maps to a zero contribution.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Sequence

from models.hypothesis import Hypothesis, Prediction

logger = logging.getLogger(__name__)


# Regex: number (int/float, optional sign and exponent)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Comparator / quantitative-language tokens
_COMPARATOR_RE = re.compile(
    r"(>=|<=|>|<|==|≥|≤|±|increase[sd]?|decrease[sd]?|reduce[sd]?|"
    r"by\s+\d|fold|times|percent|%)",
    re.IGNORECASE,
)
# Common scientific units
_UNIT_RE = re.compile(
    r"\b(\d+\.?\d*)\s*"
    r"(nm|µm|um|mm|cm|m|km|ng|µg|ug|mg|g|kg|nM|µM|uM|mM|M|"
    r"min|h|hr|hrs|day|days|sec|s|Hz|kHz|MHz|GHz|"
    r"V|mV|A|mA|W|J|cal|kcal|°C|K|Pa|kPa|MPa|"
    r"bp|kb|Mb|copies|cells|ml|µl|ul|L)\b",
    re.IGNORECASE,
)


def score_prediction(pred: Prediction) -> float:
    """Score a single structured ``Prediction`` in [0, 1]."""
    if not isinstance(pred, Prediction):
        return 0.0
    score = 0.0
    if pred.is_falsifiable():
        score += 0.5
    if pred.unit and pred.unit.strip():
        score += 0.2
    if pred.quantity and pred.quantity.strip():
        score += 0.1
    if pred.rationale and len(pred.rationale.strip()) > 10:
        score += 0.1
    if pred.ci > 0:
        score += 0.1
    return min(1.0, score)


def score_predictions(predictions: Sequence[Prediction]) -> float:
    """Mean falsifiability score over a list of ``Prediction`` objects."""
    if not predictions:
        return 0.0
    return sum(score_prediction(p) for p in predictions) / len(predictions)


def score_text_predictions(texts: Iterable[str]) -> float:
    """Heuristic scorer for free-text testable predictions.

    Each text earns up to 1.0:
      +0.4 if it contains a number,
      +0.3 if it contains a comparator/quantitative verb,
      +0.3 if it contains a recognised scientific unit.
    The hypothesis-level score is the mean across predictions.
    """
    items = [t for t in (texts or []) if t and t.strip()]
    if not items:
        return 0.0
    scores: List[float] = []
    for t in items:
        s = 0.0
        if _NUM_RE.search(t):
            s += 0.4
        if _COMPARATOR_RE.search(t):
            s += 0.3
        if _UNIT_RE.search(t):
            s += 0.3
        scores.append(min(1.0, s))
    return sum(scores) / len(scores)


def score_hypothesis(hyp: Hypothesis) -> float:
    """Combined falsifiability score for a hypothesis in [0, 1].

    If ``falsifiable_predictions`` is populated we trust it (weight 0.8) and
    take only a tiny credit (0.2) from the free-text scorer. Otherwise the
    free-text scorer carries the full weight.
    """
    structured = score_predictions(getattr(hyp, "falsifiable_predictions", []))
    free = score_text_predictions(getattr(hyp, "testable_predictions", []))

    if structured > 0:
        return 0.8 * structured + 0.2 * free
    return free


def annotate(hyp: Hypothesis) -> Hypothesis:
    """Compute and store the falsifiability score on the hypothesis in place."""
    hyp.falsifiability_score = score_hypothesis(hyp)
    return hyp


__all__ = [
    "annotate",
    "score_hypothesis",
    "score_prediction",
    "score_predictions",
    "score_text_predictions",
]
