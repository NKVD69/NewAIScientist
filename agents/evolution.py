"""
agents/evolution.py — EvolutionAgent for hypothesis refinement and improvement.

Responsible for:
- Evolving hypotheses via enhancement, simplification, divergent thinking
- LLM-powered refinement of evolution drafts
"""

from __future__ import annotations

import logging

from models.hypothesis import Hypothesis, UserFeedback
from utils import bradley_terry as bt
from utils.llm import ensure_str, get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """Refines and improves hypotheses through multiple strategies"""

    name = "Evolution"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.evolved_hypotheses = 0

    @staticmethod
    def _inherit_rating(child: Hypothesis, parent: Hypothesis) -> None:
        """Seed the offspring's rating from its parent.

        Fixes the anti-Darwinian defect: offspring used to start at the flat
        1200 default while their parents -- selected for evolution *because*
        they ranked highest -- sat above it. With ~1.7 matches each, children
        could never close that gap, so the tournament systematically preferred
        the initial generation and the evolutionary loop could produce no
        selective gain.

        Inheritance is partial (regression toward the mean) because an evolved
        hypothesis is genuinely a different object, and sigma is inflated
        because we know less about the child than about the parent. Wide error
        bars are what make the information-gain pairer prioritise it for
        testing rather than leave it unplayed.
        """
        parent_rating = bt.Rating(
            mu=parent.rating_mu,
            sigma=parent.rating_sigma,
            matches=parent.rating_matches,
        )
        child_rating = bt.inherit(parent_rating)
        child.rating_mu = child_rating.mu
        child.rating_sigma = child_rating.sigma
        child.rating_matches = 0          # the child has played nothing yet
        child.elo_rating = child_rating.mu

    @staticmethod
    def _record_refutations(child: Hypothesis, refutations: list[dict] | None) -> None:
        """Carry the refuted quantities forward as explicit limitations.

        A revised hypothesis must not silently drop the fact that its parent
        was contradicted -- otherwise the lineage launders a refutation into a
        fresh-looking hypothesis after one evolution step.
        """
        for ref in refutations or []:
            child.limitations.append(
                f"Parent refuted on {ref.get('quantity', '?')}: expected "
                f"{ref.get('expected')} {ref.get('unit', '')}, observed "
                f"{ref.get('observed')}"
            )

    @staticmethod
    def _format_refutations(refutations: list[dict] | None) -> str:
        """Render refutations for injection into the revision prompt."""
        if not refutations:
            return ""
        lines = ["", "The following pre-registered predictions were REFUTED by experiment:"]
        for ref in refutations:
            lines.append(
                f"  - {ref.get('quantity', '?')}: predicted "
                f"{ref.get('expected')} {ref.get('unit', '')}, measured "
                f"{ref.get('observed')} (deviation {ref.get('deviation')})"
            )
        lines.append("")
        lines.append(
            "Revise the mechanism so that it ACCOUNTS for these measurements. "
            "Do not simply restate the original claim with softer wording, and "
            "do not widen the thresholds to accommodate the failure -- that is "
            "post-hoc rationalisation. If the measurements cannot be reconciled "
            "with the mechanism, say so and propose a different mechanism."
        )
        return "\n".join(lines)

    async def evolve_hypothesis(self,
                               hypothesis: Hypothesis,
                               strategy: str = "enhancement",
                               refutations: list[dict] | None = None) -> Hypothesis:
        """
        Improve hypothesis using specified strategy:
        - enhancement: ground in literature
        - simplification: make clearer and more concise
        - combination: combine with other top hypotheses
        - inspiration: derive from top hypotheses
        - experimental_revision: repair the specific claims that were refuted

        ``refutations`` carries the structured verdicts from
        ``utils.adjudication`` (quantity, expected, observed, unit). Passing
        them lets the revision target the claim that actually failed, instead
        of handing the LLM an opaque blob of stdout and hoping it infers what
        went wrong.
        """
        new_hyp = Hypothesis(
            title=hypothesis.title + f" (Evolved: {strategy})",
            description=hypothesis.description,
            mechanism=hypothesis.mechanism,
            parent_ids=[hypothesis.id],
            generation_method="evolved"
        )
        self._inherit_rating(new_hyp, hypothesis)

        if strategy == "enhancement":
            new_hyp = await self._enhance_with_grounding(new_hyp, hypothesis)
        elif strategy == "simplification":
            new_hyp = await self._simplify(new_hyp, hypothesis)
        elif strategy == "out_of_box":
            new_hyp = await self._divergent_thinking(new_hyp, hypothesis)
        elif strategy == "experimental_revision":
            new_hyp.title = f"Revised: {hypothesis.title}"
            new_hyp.generation_method = "experimental-revision"
            self._record_refutations(new_hyp, refutations)

        # Try LLM-based refinement if available
        if self.llm_client:
            new_hyp = await self._llm_refine_evolution(
                new_hyp, hypothesis, strategy, refutations=refutations,
            )

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

    async def _llm_refine_evolution(
        self,
        new_hyp: Hypothesis,
        original: Hypothesis,
        strategy: str,
        refutations: list[dict] | None = None,
    ) -> Hypothesis:
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
            # Prefer the structured verdicts over the raw stdout blob: they
            # name the exact quantity that failed and by how much.
            structured = self._format_refutations(refutations)
            if structured:
                experimental_results_text = structured
            else:
                experimental_results_text = (
                    f"\n- Experimental/Simulation Results: {original.experimental_results[:1500]}\n"
                )
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
        self._inherit_rating(new_hyp, hypothesis)

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
