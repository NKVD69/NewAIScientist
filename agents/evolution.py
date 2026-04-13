from typing import List, Dict, Tuple
from models import Hypothesis, ResearchGoal, ContextMemory
import random

class RankingAgent:
    """Agent in charge of pairwise comparisons and Elo rating updates."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.matches_completed = 0

    async def conduct_match(self, h1: Hypothesis, h2: Hypothesis, goal: ResearchGoal) -> Tuple[float, float]:
        """Runs a pairwise comparison and returns the new Elo ratings."""
        print(f"  🏆 Match: {h1.title[:30]}... vs {h2.title[:30]}...")
        # Mock tournament logic
        winner = random.choice([1, 2, 0]) # 0 for draw
        
        # Elo update logic
        k = 32
        e1 = 1 / (1 + 10 ** ((h2.elo_rating - h1.elo_rating) / 400))
        e2 = 1 - e1
        s1 = 1 if winner == 1 else (0.5 if winner == 0 else 0)
        s2 = 1 - s1
        
        new_r1 = h1.elo_rating + k * (s1 - e1)
        new_r2 = h2.elo_rating + k * (s2 - e2)
        
        self.matches_completed += 1
        return new_r1, new_r2

class EvolutionAgent:
    """Agent that evolves hypotheses through mutation and crossover."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.evolved_hypotheses = 0

    async def evolve(self, parent1: Hypothesis, parent2: Optional[Hypothesis], goal: ResearchGoal) -> Hypothesis:
        """Creates a new hypothesis by combining or mutating parents."""
        print(f"  🧬 Evolving: {parent1.title[:30]}...")
        # Mock evolution
        child = Hypothesis(
            title=f"Evolved: {parent1.title}",
            description=f"Refined version based on {parent1.id}",
            mechanism=parent1.mechanism,
            novelty_level="High",
            generation_method="evolution",
            parent_id=parent1.id,
            link_type="refines"
        )
        self.evolved_hypotheses += 1
        return child
