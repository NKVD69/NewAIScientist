"""
agents/scoping.py — ScopingAgent for scientific research framing.

Responsible for:
- Analyzing state of the art from retrieved papers
- Generating structured research questions (PICO/FINER)
- Building conceptual frameworks (causal DAGs)
- Scoring questions on novelty × feasibility × impact
"""

from __future__ import annotations

import logging
from typing import Any

from models.hypothesis import (
    ResearchGoal,
    ResearchQuestion,
    StateOfArt,
)
from utils.llm import get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ScopingAgent(BaseAgent):
    """Frames the research before hypothesis generation."""

    name = "Scoping"

    # ------------------------------------------------------------------
    # STATE OF THE ART
    # ------------------------------------------------------------------

    async def analyze_state_of_art(
        self, papers: list[dict], goal: ResearchGoal
    ) -> StateOfArt:
        """Synthesize retrieved papers into known facts, gaps, contradictions."""
        if not self.llm_client or not papers:
            return self._fallback_state_of_art(papers, goal)

        summaries = "\n".join(
            [f"- {p['title']}: {p.get('summary', '')[:300]}" for p in papers[:10]]
        )

        prompt = f"""You are a senior research analyst performing a systematic scoping review.

Research Goal: {goal.title}
Domain: {goal.domain}
Description: {goal.description}

Literature found:
{summaries}

Analyze these papers and produce a structured State of the Art synthesis.
Return a JSON object with EXACTLY these keys:
{{
  "known_facts": ["Established fact 1 supported by multiple papers", "..."],
  "gaps": ["Identified gap 1 where knowledge is missing", "..."],
  "contradictions": ["Contradiction 1 between studies", "..."],
  "summary": "A concise narrative summary of the current state of knowledge (2-3 paragraphs)"
}}
Return ONLY the JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            return StateOfArt(
                known_facts=data.get("known_facts", []),
                gaps=data.get("gaps", []),
                contradictions=data.get("contradictions", []),
                summary=data.get("summary", ""),
            )
        except Exception as e:
            logger.error("State of art analysis failed: %s", e)
            return self._fallback_state_of_art(papers, goal)

    def _fallback_state_of_art(
        self, papers: list[dict], goal: ResearchGoal
    ) -> StateOfArt:
        """Simple fallback when LLM is unavailable."""
        return StateOfArt(
            known_facts=[f"Paper: {p['title']}" for p in papers[:5]],
            gaps=[f"Further research needed in {goal.domain}"],
            contradictions=[],
            summary=f"Preliminary review of {len(papers)} papers in {goal.domain}.",
        )

    # ------------------------------------------------------------------
    # RESEARCH QUESTIONS
    # ------------------------------------------------------------------

    async def generate_research_questions(
        self, state_of_art: StateOfArt, goal: ResearchGoal
    ) -> list[ResearchQuestion]:
        """Generate structured research questions from the state of the art."""
        if not self.llm_client:
            return self._fallback_questions(state_of_art, goal)

        gaps_text = "\n".join([f"- {g}" for g in state_of_art.gaps[:10]])
        facts_text = "\n".join([f"- {f}" for f in state_of_art.known_facts[:10]])
        contradictions_text = "\n".join(
            [f"- {c}" for c in state_of_art.contradictions[:5]]
        )

        prompt = f"""You are an expert at formulating precise research questions using the PICO/FINER framework.

Research Goal: {goal.title}
Domain: {goal.domain}

Known Facts:
{facts_text}

Identified Gaps:
{gaps_text}

Contradictions:
{contradictions_text}

Generate 3-5 structured research questions that address the identified gaps.
For each, indicate the type (descriptive, correlational, causal, exploratory).
If applicable to biomedical research, provide PICO components.

Return a JSON list:
[
  {{
    "question": "The full research question text",
    "type": "causal",
    "pico": {{"population": "...", "intervention": "...", "comparison": "...", "outcome": "..."}},
    "parent_gap": "The gap this question addresses"
  }}
]
Return ONLY the JSON list."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)

            if isinstance(data, dict):
                # LLM may wrap in {"questions": [...]}
                lists = [v for v in data.values() if isinstance(v, list)]
                data = lists[0] if lists else []

            questions = []
            for item in data:
                questions.append(
                    ResearchQuestion(
                        question=item.get("question", ""),
                        type=item.get("type", "exploratory"),
                        pico=item.get("pico"),
                        parent_gap=item.get("parent_gap", ""),
                    )
                )
            return questions
        except Exception as e:
            logger.error("Research question generation failed: %s", e)
            return self._fallback_questions(state_of_art, goal)

    def _fallback_questions(
        self, state_of_art: StateOfArt, goal: ResearchGoal
    ) -> list[ResearchQuestion]:
        """Deterministic fallback questions derived from gaps."""
        questions = []
        for gap in state_of_art.gaps[:3]:
            questions.append(
                ResearchQuestion(
                    question=f"What mechanisms underlie {gap}?",
                    type="exploratory",
                    parent_gap=gap,
                )
            )
        if not questions:
            questions.append(
                ResearchQuestion(
                    question=f"What are the key factors in {goal.title}?",
                    type="exploratory",
                    parent_gap="General research gap",
                )
            )
        return questions

    # ------------------------------------------------------------------
    # CONCEPTUAL FRAMEWORK
    # ------------------------------------------------------------------

    async def build_conceptual_framework(
        self, questions: list[ResearchQuestion], goal: ResearchGoal
    ) -> dict[str, Any]:
        """Build a causal DAG conceptual framework from research questions."""
        if not self.llm_client or not questions:
            return self._fallback_framework(questions, goal)

        questions_text = "\n".join(
            [f"- [{q.type}] {q.question}" for q in questions[:5]]
        )

        prompt = f"""You are an expert in research methodology and causal modeling.

Research Goal: {goal.title}
Domain: {goal.domain}

Research Questions:
{questions_text}

Build a conceptual framework as a causal DAG (Directed Acyclic Graph).
Identify key variables and their hypothesized causal relationships.

Return a JSON object:
{{
  "variables": [
    {{"name": "Variable A", "type": "independent", "description": "..."}},
    {{"name": "Variable B", "type": "dependent", "description": "..."}},
    {{"name": "Variable C", "type": "confounding", "description": "..."}}
  ],
  "edges": [
    {{"from": "Variable A", "to": "Variable B", "relationship": "positive causal", "strength": "strong"}},
    {{"from": "Variable C", "to": "Variable B", "relationship": "confounding", "strength": "moderate"}}
  ],
  "narrative": "A brief narrative description of the conceptual model."
}}
Return ONLY the JSON."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)
            return data
        except Exception as e:
            logger.error("Conceptual framework generation failed: %s", e)
            return self._fallback_framework(questions, goal)

    def _fallback_framework(
        self, questions: list[ResearchQuestion], goal: ResearchGoal
    ) -> dict[str, Any]:
        """Minimal fallback framework."""
        return {
            "variables": [
                {"name": goal.domain, "type": "independent", "description": "Primary domain"},
                {"name": "Outcome", "type": "dependent", "description": "Research outcome"},
            ],
            "edges": [
                {"from": goal.domain, "to": "Outcome", "relationship": "causal", "strength": "unknown"},
            ],
            "narrative": f"Preliminary framework for {goal.title}.",
        }

    # ------------------------------------------------------------------
    # QUESTION SCORING
    # ------------------------------------------------------------------

    async def score_questions(
        self, questions: list[ResearchQuestion]
    ) -> list[ResearchQuestion]:
        """Score each question on novelty × feasibility × impact."""
        if not self.llm_client or not questions:
            # Assign default scores
            for q in questions:
                q.novelty_score = 0.5
                q.feasibility_score = 0.5
                q.impact_score = 0.5
            return questions

        questions_text = "\n".join(
            [f"{i+1}. [{q.type}] {q.question}" for i, q in enumerate(questions)]
        )

        prompt = f"""Score each of the following research questions on three dimensions (0.0 to 1.0):
- novelty_score: How novel and original is this question?
- feasibility_score: How feasible is it to answer with current methods/resources?
- impact_score: How impactful would the answer be for the field?

Questions:
{questions_text}

Return a JSON list with ONE object per question, in the SAME order:
[
  {{"novelty_score": 0.8, "feasibility_score": 0.6, "impact_score": 0.9}},
  ...
]
Return ONLY the JSON list."""

        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=True,
            )
            data = parse_json_response(response.choices[0].message.content)

            if isinstance(data, dict):
                lists = [v for v in data.values() if isinstance(v, list)]
                data = lists[0] if lists else []

            for i, scores in enumerate(data):
                if i < len(questions):
                    questions[i].novelty_score = float(scores.get("novelty_score", 0.5))
                    questions[i].feasibility_score = float(scores.get("feasibility_score", 0.5))
                    questions[i].impact_score = float(scores.get("impact_score", 0.5))
        except Exception as e:
            logger.error("Question scoring failed: %s", e)
            for q in questions:
                q.novelty_score = 0.5
                q.feasibility_score = 0.5
                q.impact_score = 0.5

        return questions


__all__ = ["ScopingAgent"]
