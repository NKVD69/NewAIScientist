"""
agents/evolution.py — EvolutionAgent for hypothesis refinement and improvement.

Responsible for:
- Evolving hypotheses via enhancement, simplification, divergent thinking
- LLM-powered refinement of evolution drafts
"""

from __future__ import annotations

import logging

from models.hypothesis import Hypothesis, UserFeedback
from utils.llm import ensure_str, get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """Refines and improves hypotheses through multiple strategies"""

    name = "Evolution"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.evolved_hypotheses = 0

    async def evolve_hypothesis(self,
                               hypothesis: Hypothesis,
                               strategy: str = "enhancement") -> Hypothesis:
        """
        Improve hypothesis using specified strategy:
        - enhancement: ground in literature
        - simplification: make clearer and more concise
        - combination: combine with other top hypotheses
        - inspiration: derive from top hypotheses
        """
        new_hyp = Hypothesis(
            title=hypothesis.title + f" (Evolved: {strategy})",
            description=hypothesis.description,
            mechanism=hypothesis.mechanism,
            parent_ids=[hypothesis.id],
            generation_method="evolved"
        )

        if strategy == "enhancement":
            new_hyp = await self._enhance_with_grounding(new_hyp, hypothesis)
        elif strategy == "simplification":
            new_hyp = await self._simplify(new_hyp, hypothesis)
        elif strategy == "out_of_box":
            new_hyp = await self._divergent_thinking(new_hyp, hypothesis)
        elif strategy == "experimental_revision":
            new_hyp.title = f"Revised: {hypothesis.title}"
            new_hyp.generation_method = "experimental-revision"

        # Try LLM-based refinement if available
        if self.llm_client:
            new_hyp = await self._llm_refine_evolution(new_hyp, hypothesis, strategy)

        self.evolved_hypotheses += 1
        return new_hyp

    async def _enhance_with_grounding(self, new_hyp: Hypothesis,
                                     original: Hypothesis) -> Hypothesis:
        new_hyp.mechanism = (
            f"Enhanced mechanism: {original.mechanism} "
            f"Additionally grounded by identifying supporting molecular pathways "
            f"and experimental evidence from recent literature."
        )
        new_hyp.grounding_evidence = original.grounding_evidence + [
            "Additional pathway analysis",
            "Cross-validation against recent meta-analyses"
        ]
        new_hyp.testable_predictions = original.testable_predictions + [
            "Advanced prediction: Multi-dimensional experimental validation",
        ]
        return new_hyp

    async def _simplify(self, new_hyp: Hypothesis,
                       original: Hypothesis) -> Hypothesis:
        new_hyp.title = f"Simplified: {original.title}"
        new_hyp.mechanism = (
            "Core simplified mechanism: "
            + original.mechanism.split('.')[0] + ". "
            + "Reduces complexity by focusing on primary pathway."
        )
        new_hyp.testable_predictions = original.testable_predictions[:2]
        new_hyp.limitations = original.limitations + [
            "Simplified version may miss secondary effects"
        ]
        return new_hyp

    async def _divergent_thinking(self, new_hyp: Hypothesis,
                                 original: Hypothesis) -> Hypothesis:
        new_hyp.title = f"Divergent: {original.title}"
        new_hyp.description = (
            f"Exploring lateral connections and unorthodox pathways inspired by: {original.title}. "
            f"This hypothesis aggressively seeks to bridge disconnected domains."
        )
        new_hyp.mechanism = (
            "Divergent mechanism: Re-evaluating the core assumptions. Applying principles from "
            "far-field disciplines (e.g., astrophysics, ecology, computer science) to the target domain."
        )
        return new_hyp

    async def _llm_refine_evolution(self, new_hyp: Hypothesis, original: Hypothesis, strategy: str) -> Hypothesis:
        """Use LLM to refine evolved hypothesis"""
        experimental_results_text = ""
        if strategy == "out_of_box":
            system_prompt = "You are a visionary scientist specializing in 'lateral thinking'. Your task is to force a radical, cross-disciplinary jump."
            task_instruction = (
                "Completely ignore the dominant paradigm of the Original Hypothesis. "
                "Find a mechanism from a totally unrelated scientific field and boldly apply it to this problem."
            )
        elif strategy == "experimental_revision":
            system_prompt = "You are a rigorous scientist revising a hypothesis in light of empirical evidence."
            task_instruction = (
                "Review the original hypothesis and the empirical/experimental results. "
                "Modify the hypothesis description, biochemical/physical mechanism, and predictions to reconcile "
                "them with the experimental findings (especially addressing and adjusting any refuted predictions)."
            )
            experimental_results_text = f"\n- Experimental/Simulation Results: {original.experimental_results}\n"
        else:
            system_prompt = "You are a meticulous scientific research assistant."
            task_instruction = f"Improve the following hypothesis using the '{strategy}' strategy. Ground it in realistic pathways."

        prompt = f"""{system_prompt}

{task_instruction}

Original Hypothesis:
- Title: {original.title}
- Mechanism: {original.mechanism}
- Description: {original.description}
- Testable Predictions: {original.testable_predictions}{experimental_results_text}

Current Evolution Draft:
- Title: {new_hyp.title}
- Mechanism: {new_hyp.mechanism}

Provide an improved version as a JSON object with keys: "title", "description", "mechanism", "testable_predictions" (list of strings), "limitations" (list of strings).
**IMPORTANT: Output ONLY the raw JSON object.** Do NOT wrap it in markdown block quotes.
"""
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                json_mode=True
            )

            data = parse_json_response(response.choices[0].message.content)
            new_hyp.title = ensure_str(data.get("title", new_hyp.title))
            new_hyp.description = ensure_str(data.get("description", new_hyp.description))
            new_hyp.mechanism = ensure_str(data.get("mechanism", new_hyp.mechanism))
            new_hyp.testable_predictions = data.get("testable_predictions", new_hyp.testable_predictions)
            new_hyp.limitations = data.get("limitations", new_hyp.limitations)
            if strategy == "experimental_revision":
                new_hyp.generation_method = "experimental-revision"
            else:
                new_hyp.generation_method = "evolved-llm"
        except Exception as e:
            logger.warning("LLM evolution refinement failed: %s", e)

        return new_hyp


    async def evolve_with_feedback(
        self,
        hypothesis: Hypothesis,
        feedback: UserFeedback,
    ) -> Hypothesis | None:
        """Refine a hypothesis using a scientist's structured feedback.

        Behaviour by ``feedback.verdict``:
        - ``"agree"``    → no change; returns ``None``.
        - ``"disagree"`` → returns ``None`` (caller should drop the hypothesis).
        - ``"refine"``   → returns a new evolved hypothesis whose mechanism
          and predictions incorporate the comment as an explicit constraint.
        """
        verdict = feedback.verdict
        if verdict == "agree":
            self._note_feedback(hypothesis, feedback)
            return None
        if verdict == "disagree":
            self._note_feedback(hypothesis, feedback)
            return None

        # verdict == "refine"
        new_hyp = Hypothesis(
            title=f"{hypothesis.title} (refined)",
            description=hypothesis.description,
            mechanism=hypothesis.mechanism,
            testable_predictions=list(hypothesis.testable_predictions),
            limitations=list(hypothesis.limitations),
            grounding_evidence=list(hypothesis.grounding_evidence),
            cited_papers=list(hypothesis.cited_papers),
            parent_ids=[hypothesis.id],
            generation_method="evolved-feedback",
        )

        # Always record the human constraint, even when the LLM is offline.
        constraint = (feedback.comment or "").strip()
        if constraint:
            new_hyp.limitations.append(f"Scientist constraint: {constraint}")

        if self.llm_client and constraint:
            new_hyp = await self._llm_refine_with_feedback(new_hyp, hypothesis, constraint)

        self.evolved_hypotheses += 1
        self._note_feedback(new_hyp, feedback)
        return new_hyp

    @staticmethod
    def _note_feedback(hyp: Hypothesis, feedback: UserFeedback) -> None:
        """Stash the feedback on the hypothesis for later inspection/export."""
        existing = getattr(hyp, "_user_feedback", None)
        if existing is None:
            hyp._user_feedback = [feedback]  # type: ignore[attr-defined]
        else:
            existing.append(feedback)

    async def _llm_refine_with_feedback(
        self,
        new_hyp: Hypothesis,
        original: Hypothesis,
        constraint: str,
    ) -> Hypothesis:
        """LLM call that injects the scientist's free-text critique as a hard constraint."""
        prompt = f"""You are a senior scientific collaborator refining a hypothesis based on
direct feedback from the principal investigator.

Original hypothesis:
- Title: {original.title}
- Description: {original.description}
- Mechanism: {original.mechanism}
- Testable predictions: {original.testable_predictions}

The scientist's critique (treat this as a hard constraint, NOT a suggestion):
"\"\"\"
{constraint}
\"\"\""

Produce a refined hypothesis that addresses the critique without losing the
original's scientific intent. Output ONLY a raw JSON object with keys:
"title", "description", "mechanism", "testable_predictions" (list of strings),
"limitations" (list of strings)."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            new_hyp.title = ensure_str(data.get("title", new_hyp.title))
            new_hyp.description = ensure_str(data.get("description", new_hyp.description))
            new_hyp.mechanism = ensure_str(data.get("mechanism", new_hyp.mechanism))
            new_hyp.testable_predictions = data.get(
                "testable_predictions", new_hyp.testable_predictions,
            )
            new_hyp.limitations = data.get("limitations", new_hyp.limitations)
        except Exception as e:  # noqa: BLE001
            logger.warning("Feedback-driven LLM refinement failed: %s", e)
        return new_hyp


__all__ = ["EvolutionAgent"]
