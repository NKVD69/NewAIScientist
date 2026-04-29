"""
agents/reflection.py — ReflectionAgent for hypothesis review and critique.

Responsible for:
- Multi-dimensional hypothesis evaluation (correctness, novelty, testability, quality)
- LLM-powered reviews with structured feedback
- Simulated review fallback
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from models.hypothesis import Hypothesis, ResearchGoal, ReviewCritique
from utils.llm import get_llm_completion, parse_json_response, ensure_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """Reviews hypotheses for correctness, quality, novelty, testability"""

    name = "Reflection"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.reviews_completed = 0

    async def review_hypothesis(self, 
                                hypothesis: Hypothesis,
                                goal: ResearchGoal) -> ReviewCritique:
        """
        Comprehensive hypothesis review.
        Uses LLM if available, otherwise falls back to simulation.
        """
        
        if self.llm_client:
            try:
                review = await self._review_with_llm(hypothesis, goal)
                if review:
                    hypothesis.reviews.append(review)
                    self._update_novelty_level(hypothesis, review.novelty_score)
                    self.reviews_completed += 1
                    return review
            except Exception as e:
                logger.warning("LLM review failed: %s. Falling back to simulation.", e)

        # Fallback to simulated review
        return await self._review_simulated(hypothesis, goal)

    async def _review_with_llm(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ReviewCritique:
        """Perform review using local LLM"""
        prompt = self._build_review_prompt(hypothesis, goal)
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True
            )
        except Exception as e:
            logger.warning("LLM review API call failed: %s", e)
            raise e
        
        data = parse_json_response(response.choices[0].message.content)
        
        return ReviewCritique(
            review_type="llm-full",
            correctness_score=float(data.get("correctness_score", 0.5)),
            novelty_score=float(data.get("novelty_score", 0.5)),
            testability_score=float(data.get("testability_score", 0.5)),
            quality_score=float(data.get("quality_score", 0.5)),
            feedback=data.get("feedback", "No feedback provided.")
        )

    def _build_review_prompt(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        return f"""
You are a senior scientific reviewer. Your task is to critically evaluate the following research hypothesis.

**Research Goal:** {goal.title} ({goal.domain})
**Hypothesis:** {hypothesis.title}
**Description:** {hypothesis.description}
**Mechanism:** {hypothesis.mechanism}
**Predictions:** {', '.join(hypothesis.testable_predictions)}

Evaluate this hypothesis on the following criteria and return a JSON object:
1. **correctness_score** (0.0-1.0): Is it scientifically sound and logically consistent?
2. **novelty_score** (0.0-1.0): Is it new compared to established knowledge?
3. **testability_score** (0.0-1.0): Can it be validated experimentally?
4. **quality_score** (0.0-1.0): Overall assessment.
5. **feedback**: A concise paragraph of constructive criticism.

**JSON Format:**
{{
  "correctness_score": 0.8,
  "novelty_score": 0.7,
  "testability_score": 0.9,
  "quality_score": 0.8,
  "feedback": "The hypothesis is..."
}}
"""

    def _update_novelty_level(self, hypothesis: Hypothesis, score: float):
        if score > 0.75:
            hypothesis.novelty_level = "very_high"
        elif score > 0.55:
            hypothesis.novelty_level = "high"
        elif score > 0.35:
            hypothesis.novelty_level = "medium"
        else:
            hypothesis.novelty_level = "low"

    async def _review_simulated(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ReviewCritique:
        """Simulated review logic (fallback)"""
        correctness = await self._assess_correctness(hypothesis, goal)
        novelty = await self._assess_novelty(hypothesis, goal)
        testability = await self._assess_testability(hypothesis, goal)
        quality = self._compute_quality_score(correctness, novelty, testability)
        
        self._update_novelty_level(hypothesis, novelty)
        
        feedback = self._generate_review_feedback(
            correctness, novelty, testability, quality, hypothesis
        )
        
        review = ReviewCritique(
            review_type="simulated-full",
            correctness_score=correctness,
            novelty_score=novelty,
            testability_score=testability,
            quality_score=quality,
            feedback=feedback
        )
        
        hypothesis.reviews.append(review)
        self.reviews_completed += 1
        
        return review
    
    async def _assess_correctness(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        score = 0.7
        if hypothesis.grounding_evidence:
            score += 0.15
        if hypothesis.limitations:
            score -= len(hypothesis.limitations) * 0.05
        return min(1.0, max(0.0, score))
    
    async def _assess_novelty(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        score = 0.6
        if "simulated" in hypothesis.generation_method:
            score = 0.55
        elif "llm" in hypothesis.generation_method:
            score = 0.75
        elif hypothesis.generation_method == "evolved":
            score = 0.65
        elif hypothesis.generation_method == "combined":
            score = 0.70
        elif hypothesis.generation_method == "inspired":
            score = 0.60
        if "similar to" in hypothesis.description.lower():
            score -= 0.2
        return min(1.0, max(0.0, score))
    
    async def _assess_testability(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        score = 0.65
        if len(hypothesis.testable_predictions) >= 3:
            score += 0.2
        elif len(hypothesis.testable_predictions) >= 1:
            score += 0.1
        if len(hypothesis.mechanism) > 50:
            score += 0.1
        if "requires novel techniques" in str(hypothesis.limitations):
            score -= 0.15
        return min(1.0, max(0.0, score))
    
    def _compute_quality_score(self, correctness: float, novelty: float, testability: float) -> float:
        quality = (correctness * 0.4 + novelty * 0.3 + testability * 0.3)
        return min(1.0, max(0.0, quality))

    async def _assess_quality(self, hypothesis: Hypothesis, goal: ResearchGoal) -> float:
        correctness = await self._assess_correctness(hypothesis, goal)
        novelty = await self._assess_novelty(hypothesis, goal)
        testability = await self._assess_testability(hypothesis, goal)
        return self._compute_quality_score(correctness, novelty, testability)
    
    def _generate_review_feedback(self, correctness: float, novelty: float, 
                                 testability: float, quality: float,
                                 hypothesis: Hypothesis) -> str:
        feedback_parts = []
        if correctness > 0.8:
            feedback_parts.append("✓ Logically sound and well-grounded")
        elif correctness < 0.5:
            feedback_parts.append("⚠ Logical consistency concerns identified")
        if novelty > 0.7:
            feedback_parts.append("✓ Proposes genuinely novel elements")
        elif novelty < 0.4:
            feedback_parts.append("⚠ Limited novelty over existing literature")
        if testability > 0.7:
            feedback_parts.append("✓ Clear testable predictions")
        elif testability < 0.5:
            feedback_parts.append("⚠ May require refinement for experimental validation")
        if quality > 0.75:
            feedback_parts.append("✓ High quality overall")
        return " | ".join(feedback_parts) if feedback_parts else "Further review recommended"


__all__ = ["ReflectionAgent"]
