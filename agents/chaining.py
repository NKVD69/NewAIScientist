import logging

from models import Hypothesis, ResearchGoal

logger = logging.getLogger(__name__)

class HypothesisChainingAgent:
    """
    Agent responsible for identifying logical links between hypotheses
    to build multi-step research directions (Chains of Thought).
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.chains_discovered = 0

    async def identify_links(self, hypotheses: list[Hypothesis], goal: ResearchGoal) -> list[dict]:
        """
        Analyzes a set of hypotheses and identifies dependencies or refinement links.
        """
        logger.info("ChainingAgent is looking for connections among %d hypotheses...", len(hypotheses))

        # Simplified logic for finding links
        links = []
        if len(hypotheses) >= 2:
            # Example: H2 builds on H1
            links.append({
                "source": hypotheses[0].id,
                "target": hypotheses[1].id,
                "type": "refines",
                "reason": "H2 specifies the molecular sub-target identified in H1's broader pathway."
            })

        self.chains_discovered += len(links)
        return links

    async def synthesize_chain(self, chain: list[Hypothesis]) -> str:
        """
        Synthesizes a master narrative for a chain of hypotheses.
        """
        narrative = " -> ".join([h.title for h in chain])
        return f"Research Chain: {narrative}"
