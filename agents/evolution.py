"""
agents/evolution.py — EvolutionAgent for hypothesis refinement and improvement.

Responsible for:
- Evolving hypotheses via enhancement, simplification, divergent thinking
- LLM-powered refinement of evolution drafts
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from models.hypothesis import Hypothesis
from utils.llm import get_llm_completion, parse_json_response, ensure_str
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
        if strategy == "out_of_box":
            system_prompt = "You are a visionary scientist specializing in 'lateral thinking'. Your task is to force a radical, cross-disciplinary jump."
            task_instruction = (
                "Completely ignore the dominant paradigm of the Original Hypothesis. "
                "Find a mechanism from a totally unrelated scientific field and boldly apply it to this problem."
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
            new_hyp.generation_method = "evolved-llm"
        except Exception as e:
            logger.warning("LLM evolution refinement failed: %s", e)
        
        return new_hyp


__all__ = ["EvolutionAgent"]
