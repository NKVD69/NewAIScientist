"""
agents/reflection.py — ReflectionAgent for hypothesis review and critique.

Responsible for:
- Multi-dimensional hypothesis evaluation (correctness, novelty, testability, quality)
- Multi-agent review committee (3 specialized reviewers + 1 meta-reviewer)
- Entity validation (genes, drugs, proteins) injected into review context
- LLM-powered reviews with structured feedback
- Simulated review fallback
"""

from __future__ import annotations

import asyncio
import logging

from models.hypothesis import Hypothesis, ResearchGoal, ReviewCritique
from utils.llm import get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reviewer personas for the multi-agent review committee
# ---------------------------------------------------------------------------

_REVIEWER_PERSONAS = {
    "biologist": {
        "name": "Reviewer 1 (Biologist/Biochemist)",
        "system": (
            "You are a senior biologist and biochemist with 20 years of experience "
            "in molecular biology, protein science, and cellular mechanisms. "
            "You focus on evaluating the biological plausibility of proposed "
            "mechanisms: are the protein targets real? Are the signaling pathways "
            "correctly described? Is the proposed mechanism of action consistent "
            "with known biology?"
        ),
    },
    "pharmacologist": {
        "name": "Reviewer 2 (Translational/Pharmacologist)",
        "system": (
            "You are a translational researcher and pharmacologist specialized "
            "in drug development, ADMET, and clinical feasibility. You evaluate: "
            "Can the proposed intervention reach its target in vivo? Are the "
            "concentrations realistic? What are the likely toxicity and "
            "off-target effects? Is there a path from bench to bedside?"
        ),
    },
    "statistician": {
        "name": "Reviewer 3 (Statistician/Methodologist)",
        "system": (
            "You are a biostatistician and research methodologist. You focus on: "
            "Are the testable predictions quantifiable? What statistical tests "
            "would be appropriate? What are the potential confounders and biases? "
            "Is the proposed sample size feasible? Could the hypothesis be "
            "falsified with available methods?"
        ),
    },
}


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

    # ------------------------------------------------------------------
    # Entity validation (injected into review context)
    # ------------------------------------------------------------------

    async def _validate_hypothesis_entities(self, hypothesis: Hypothesis) -> str:
        """Run entity validation on the hypothesis text and return a formatted report."""
        try:
            from utils.experiment_sandbox import (
                format_entity_report,
                validate_entities,
            )
            text = " ".join([
                hypothesis.title or "",
                hypothesis.description or "",
                hypothesis.mechanism or "",
                " ".join(hypothesis.testable_predictions or []),
            ])
            entity_results = await validate_entities(
                text,
                timeout=5.0,
                llm_client=self.llm_client,
            )
            return format_entity_report(entity_results)
        except Exception as exc:
            logger.warning("Entity validation during review failed: %s", exc)
            return "Entity validation could not be performed."

    # ------------------------------------------------------------------
    # Multi-Agent Review Committee
    # ------------------------------------------------------------------

    async def _review_with_llm(self, hypothesis: Hypothesis, goal: ResearchGoal) -> ReviewCritique:
        """Perform review using multi-agent review committee.

        Pipeline:
        1. Validate entities (genes, drugs, proteins) mentioned in the hypothesis.
        2. Run 3 specialized reviewers in parallel (biologist, pharmacologist, statistician).
        3. Run a meta-reviewer that synthesizes the 3 reviews + entity report into
           a final consolidated critique with scores.
        """
        # Step 1: Entity validation
        entity_report = await self._validate_hypothesis_entities(hypothesis)
        logger.info("Entity validation completed for: %s", hypothesis.title[:50])

        # Step 2: Parallel specialized reviews
        reviewer_tasks = []
        for persona_key, persona in _REVIEWER_PERSONAS.items():
            reviewer_tasks.append(
                self._run_single_reviewer(hypothesis, goal, persona, entity_report)
            )

        reviewer_results = await asyncio.gather(*reviewer_tasks, return_exceptions=True)

        # Collect successful reviews
        critiques = []
        for i, result in enumerate(reviewer_results):
            persona_key = list(_REVIEWER_PERSONAS.keys())[i]
            if isinstance(result, Exception):
                logger.warning("Reviewer '%s' failed: %s", persona_key, result)
                critiques.append(f"[{_REVIEWER_PERSONAS[persona_key]['name']}]: Review failed — {result}")
            else:
                critiques.append(result)

        # Step 3: Meta-reviewer consolidation
        return await self._run_meta_reviewer(hypothesis, goal, entity_report, critiques)

    async def _run_single_reviewer(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
        persona: dict,
        entity_report: str,
    ) -> str:
        """Run a single specialized reviewer and return its critique text."""
        prompt = f"""{persona['system']}

You are reviewing a scientific hypothesis as part of a peer review committee.

**Research Goal:** {goal.title} ({goal.domain})
**Hypothesis:** {hypothesis.title}
**Description:** {hypothesis.description}
**Mechanism:** {hypothesis.mechanism}
**Testable Predictions:** {', '.join(hypothesis.testable_predictions)}

**Entity Validation Report (automated):**
{entity_report}

As {persona['name']}, provide a detailed critique of this hypothesis from your area of expertise.
Focus on:
1. Strengths of the hypothesis from your perspective
2. Weaknesses and concerns specific to your domain
3. Suggestions for improvement
4. Whether the entity validation raises any red flags

Keep your critique focused, specific, and constructive. Write 150-300 words."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                json_mode=False,
            )
            text = response.choices[0].message.content.strip()
            return f"[{persona['name']}]:\n{text}"
        except Exception as e:
            raise RuntimeError(f"Reviewer {persona['name']} API call failed: {e}") from e

    async def _run_meta_reviewer(
        self,
        hypothesis: Hypothesis,
        goal: ResearchGoal,
        entity_report: str,
        critiques: list[str],
    ) -> ReviewCritique:
        """Meta-reviewer: consolidate all reviews into a final scored critique."""
        all_critiques = "\n\n---\n\n".join(critiques)

        prompt = f"""You are the Meta-Reviewer — a senior editor consolidating the opinions of three specialized peer reviewers.

**Research Goal:** {goal.title} ({goal.domain})
**Hypothesis:** {hypothesis.title}
**Description:** {hypothesis.description}
**Mechanism:** {hypothesis.mechanism}
**Predictions:** {', '.join(hypothesis.testable_predictions)}

**Entity Validation Report:**
{entity_report}

**Individual Reviewer Critiques:**
{all_critiques}

Based on ALL the above, synthesize a final consolidated review. Return a JSON object:
{{
  "correctness_score": 0.0-1.0,  // Scientific soundness and logical consistency
  "novelty_score": 0.0-1.0,      // Novelty compared to established knowledge
  "testability_score": 0.0-1.0,  // Can it be validated experimentally?
  "quality_score": 0.0-1.0,      // Overall quality assessment
  "entity_confidence": 0.0-1.0,  // Confidence based on entity validation results
  "feedback": "Consolidated critique paragraph synthesizing all reviews and entity validation...",
  "key_strengths": ["strength 1", "strength 2"],
  "key_weaknesses": ["weakness 1", "weakness 2"],
  "improvement_suggestions": ["suggestion 1", "suggestion 2"]
}}

Adjust scores DOWN if critical entities (genes, drugs) could not be verified.
Return ONLY the JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
        except Exception as e:
            logger.warning("Meta-reviewer API call failed: %s", e)
            raise e

        data = parse_json_response(response.choices[0].message.content)

        # Build enriched feedback with entity confidence
        feedback = data.get("feedback", "No feedback provided.")
        entity_confidence = float(data.get("entity_confidence", 0.5))

        strengths = data.get("key_strengths", [])
        weaknesses = data.get("key_weaknesses", [])
        suggestions = data.get("improvement_suggestions", [])

        # Append structured info to feedback
        if strengths:
            feedback += "\n\n**Strengths:** " + "; ".join(strengths)
        if weaknesses:
            feedback += "\n**Weaknesses:** " + "; ".join(weaknesses)
        if suggestions:
            feedback += "\n**Suggestions:** " + "; ".join(suggestions)
        feedback += f"\n**Entity Confidence:** {entity_confidence:.2f}"

        return ReviewCritique(
            review_type="multi-agent-committee",
            correctness_score=float(data.get("correctness_score", 0.5)),
            novelty_score=float(data.get("novelty_score", 0.5)),
            testability_score=float(data.get("testability_score", 0.5)),
            quality_score=float(data.get("quality_score", 0.5)),
            feedback=feedback
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
        """Assess novelty against the literature, not against plumbing.

        The previous implementation returned a value keyed on
        ``generation_method``: 0.75 for "llm-generated", 0.55 for "simulated",
        0.65 for "evolved". A hypothesis was judged 36% more novel for having
        come from an LLM rather than the simulation stub. That artefact then
        carried weight 0.25 in the Bradley-Terry prior and helped decide which
        hypothesis got written up.

        Novelty is not introspectable. It is a claim about what already exists
        in the literature, so it is answered by searching — see
        ``utils.novelty``, which runs a Semantic Scholar prior-art query and
        returns the nearest papers so a human can check the verdict.

        Returns 0.5 with ``novelty_level = "unknown"`` when no search could
        run: explicit ignorance rather than a plausible-looking number.
        """
        if hypothesis.novelty_report:
            return float(hypothesis.novelty_report.get("score", 0.5))

        try:
            from utils.novelty import apply_report, assess_novelty

            report = await assess_novelty(
                hypothesis,
                rag_engine=getattr(self, "rag_engine", None),
                graph_agent=getattr(self, "graph_agent", None),
            )
            apply_report(hypothesis, report)
            if not report.searched:
                logger.info(
                    "Novelty for '%s' not assessed (%s) — recorded as unknown.",
                    (hypothesis.title or "")[:40], report.error,
                )
            return report.score
        except Exception as exc:  # noqa: BLE001
            logger.warning("Novelty assessment unavailable: %s", exc)
            hypothesis.novelty_level = "unknown"
            return 0.5

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
