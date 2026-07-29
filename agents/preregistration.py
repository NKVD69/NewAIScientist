"""
agents/preregistration.py — PreregistrationAgent for formal prediction registration.

Responsible for:
- Translating free-text testable_predictions into structured Prediction objects
- Computing falsifiability scores per hypothesis
- Creating immutable, timestamped prediction bundles (SHA-256 hash)
- Ensuring predictions are locked before experimentation

This agent bridges the gap between qualitative hypothesis statements and
quantitative, falsifiable predictions — a key requirement for rigorous science.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

from models.hypothesis import Hypothesis, Prediction, ResearchGoal
from utils.llm import get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class PreregistrationAgent(BaseAgent):
    """Formalizes free-text predictions into structured, falsifiable predictions."""

    name = "Preregistration"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.predictions_formalized = 0

    async def formalize_predictions(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
    ) -> list[Prediction]:
        """Translate free-text testable_predictions into structured Predictions.

        Each Prediction includes:
        - quantity: what is being measured
        - expected_value: the predicted value
        - ci: confidence interval (±)
        - unit: measurement unit
        - refuting_threshold: value beyond which the prediction is refuted
        - rationale: why this value is expected

        Returns the list of formalized Predictions (also stored on the hypothesis).
        """
        if not hypothesis.testable_predictions:
            logger.info(
                "No testable predictions to formalize for '%s'.",
                hypothesis.title[:50],
            )
            return []

        if self.llm_client:
            try:
                predictions = await self._formalize_with_llm(hypothesis, goal)
                if predictions:
                    hypothesis.falsifiable_predictions = predictions
                    hypothesis.falsifiability_score = self._compute_falsifiability_score(
                        predictions
                    )
                    hypothesis.prediction_hash = self._compute_prediction_hash(
                        predictions
                    )
                    hypothesis.registered_at = datetime.now().isoformat()
                    self.predictions_formalized += len(predictions)
                    logger.info(
                        "Formalized %d predictions for '%s' (score=%.2f, hash=%s).",
                        len(predictions),
                        hypothesis.title[:40],
                        hypothesis.falsifiability_score,
                        hypothesis.prediction_hash[:12],
                    )
                    return predictions
            except Exception as e:
                logger.warning("LLM prediction formalization failed: %s", e)

        # Fallback: create minimal Prediction stubs
        return self._fallback_formalize(hypothesis)

    async def _formalize_with_llm(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
    ) -> list[Prediction]:
        """Use LLM to translate qualitative predictions into quantitative ones."""
        predictions_text = "\n".join(
            [f"  {i+1}. {p}" for i, p in enumerate(hypothesis.testable_predictions)]
        )

        prompt = f"""You are a research methodologist specializing in pre-registration of scientific studies.

Research Goal: {goal.title} ({goal.domain})
Hypothesis: {hypothesis.title}
Mechanism: {hypothesis.mechanism}

Free-text predictions to formalize:
{predictions_text}

For EACH prediction above, produce a structured, quantitative pre-registration entry.
If a prediction is inherently qualitative (e.g., "gene X will be upregulated"),
estimate a plausible quantitative threshold based on the domain.

Return a JSON list:
[
  {{
    "quantity": "What is being measured (e.g., 'IC50 of Drug X on MOLM-13 cells')",
    "expected_value": 5.0,
    "ci": 2.0,
    "unit": "µM",
    "refuting_threshold": 15.0,
    "rationale": "Based on literature showing typical IC50 values in this range for kinase inhibitors"
  }}
]

Rules:
- expected_value must be a number (best estimate)
- ci is the symmetric ±CI around the expected value (95% confidence)
- refuting_threshold is the absolute distance from expected_value beyond which the prediction is REFUTED
- refuting_threshold should always be > ci (otherwise the prediction is trivially refutable)
- Be realistic with units and values for the domain ({goal.domain})

Return ONLY the JSON list."""

        response = await get_llm_completion(
            self.llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            json_mode=True,
        )
        data = parse_json_response(response.choices[0].message.content)

        # Handle LLM wrapping in a dict
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else []

        predictions = []
        for item in data:
            try:
                predictions.append(
                    Prediction(
                        quantity=str(item.get("quantity", "")),
                        expected_value=float(item.get("expected_value", 0.0)),
                        ci=float(item.get("ci", 0.0)),
                        unit=str(item.get("unit", "")),
                        refuting_threshold=float(
                            item.get("refuting_threshold", 0.0)
                        ),
                        rationale=str(item.get("rationale", "")),
                    )
                )
            except (ValueError, TypeError) as e:
                logger.debug("Skipping malformed prediction: %s", e)

        return predictions

    def _fallback_formalize(self, hypothesis: Hypothesis) -> list[Prediction]:
        """Create minimal Prediction stubs from free-text predictions."""
        predictions = []
        for text in hypothesis.testable_predictions:
            predictions.append(
                Prediction(
                    quantity=text,
                    expected_value=0.0,
                    ci=0.0,
                    unit="arbitrary",
                    refuting_threshold=0.0,
                    rationale="Stub — LLM formalization unavailable.",
                )
            )
        hypothesis.falsifiable_predictions = predictions
        hypothesis.falsifiability_score = 0.0
        hypothesis.prediction_hash = self._compute_prediction_hash(predictions)
        hypothesis.registered_at = datetime.now().isoformat()
        self.predictions_formalized += len(predictions)
        return predictions

    @staticmethod
    def _compute_falsifiability_score(predictions: list[Prediction]) -> float:
        """Compute a 0-1 falsifiability score for a bundle of predictions.

        Score is the fraction of predictions that are truly falsifiable
        (i.e., have a non-trivial refuting threshold and CI).
        """
        if not predictions:
            return 0.0
        falsifiable_count = sum(1 for p in predictions if p.is_falsifiable())
        return falsifiable_count / len(predictions)

    @staticmethod
    def _compute_prediction_hash(predictions: list[Prediction]) -> str:
        """Compute a SHA-256 hash of the prediction bundle for integrity.

        This hash is computed BEFORE experimentation and verified afterwards
        to ensure predictions were not modified post-hoc (anti-HARKing).

        The bundle contains ONLY prediction content. A previous version also
        hashed ``datetime.now()``, which made the digest differ on every call
        and left ``verify_integrity()`` returning False unconditionally — the
        anti-HARKing guarantee was vacuous. The registration timestamp is now
        stored alongside the hash (``registered_at``) instead of inside it.

        Predictions are sorted by quantity so that a reordering of the list —
        which changes nothing scientifically — does not read as tampering.
        """
        bundle = {
            "schema": "newaisci.prediction-bundle.v1",
            "predictions": sorted(
                (
                    {
                        "quantity": p.quantity,
                        "expected_value": p.expected_value,
                        "ci": p.ci,
                        "unit": p.unit,
                        "refuting_threshold": p.refuting_threshold,
                    }
                    for p in predictions
                ),
                key=lambda d: (d["quantity"], d["unit"]),
            ),
        }
        canonical = json.dumps(bundle, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def verify_integrity(self, hypothesis: Hypothesis) -> bool:
        """Check that the prediction hash still matches the stored predictions.

        Returns True iff the predictions are byte-identical (modulo ordering)
        to those registered before experimentation. Must be called AFTER the
        experiment and BEFORE the verdicts are trusted: a hypothesis whose
        predictions moved between registration and adjudication has been
        HARKed and its verdicts are worthless.
        """
        ok, _ = self.check_integrity(hypothesis)
        return ok

    @classmethod
    def check_integrity(cls, hypothesis: Hypothesis) -> tuple[bool, str]:
        """Like :meth:`verify_integrity` but explains *why* it failed.

        A bare False is unactionable — "never registered" and "modified after
        the fact" have opposite implications and must be distinguishable.
        """
        if not hypothesis.falsifiable_predictions:
            return False, "predictions were never registered"
        if not hypothesis.prediction_hash:
            return False, "predictions exist but were never sealed with a hash"

        current = cls._compute_prediction_hash(hypothesis.falsifiable_predictions)
        if current == hypothesis.prediction_hash:
            return True, f"intact (sha256:{current[:12]})"
        return False, (
            f"TAMPERED: predictions changed since registration "
            f"(registered sha256:{hypothesis.prediction_hash[:12]}, "
            f"current sha256:{current[:12]})"
        )


__all__ = ["PreregistrationAgent"]
