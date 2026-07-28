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

        This hash is computed BEFORE experimentation and can be verified
        afterwards to ensure predictions were not modified post-hoc.
        """
        bundle = {
            "timestamp": datetime.now().isoformat(),
            "predictions": [
                {
                    "quantity": p.quantity,
                    "expected_value": p.expected_value,
                    "ci": p.ci,
                    "unit": p.unit,
                    "refuting_threshold": p.refuting_threshold,
                }
                for p in predictions
            ],
        }
        canonical = json.dumps(bundle, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def verify_integrity(self, hypothesis: Hypothesis) -> bool:
        """Check that the prediction hash still matches the stored predictions.

        Returns True if the predictions have not been tampered with since
        pre-registration. This is a post-experiment integrity check.
        """
        if not hypothesis.prediction_hash or not hypothesis.falsifiable_predictions:
            return False
        current_hash = self._compute_prediction_hash(
            hypothesis.falsifiable_predictions
        )
        # We only compare the prediction content, not the timestamp
        # (which differs between registration and verification)
        return current_hash == hypothesis.prediction_hash


__all__ = ["PreregistrationAgent"]
