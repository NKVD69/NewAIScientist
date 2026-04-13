from typing import List, Dict, Optional
from models import Hypothesis, ResearchGoal

class GenerationAgent:
    """Agent specialized in generating novel scientific hypotheses."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.generated_count = 0

    async def generate_batch(self, goal: ResearchGoal, literature: List[Dict], count: int = 5) -> List[Hypothesis]:
        """Generates a batch of novel hypotheses."""
        print(f"  💡 Generating {count} hypotheses...")
        # Mock generation logic
        hyps = []
        for i in range(count):
            hyps.append(Hypothesis(
                title=f"Sample Hypothesis {self.generated_count + i}",
                description="Synthesized from literature synthesis...",
                mechanism="Activation of pathway X via ligand Y",
                novelty_level="Medium",
                generation_method="generation"
            ))
        self.generated_count += count
        return hyps

class ReflectionAgent:
    """Agent that performs peer-review style reflections on hypotheses."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.reviews_completed = 0

    async def review(self, hypothesis: Hypothesis, goal: ResearchGoal) -> Dict:
        """Provides a detailed review of a single hypothesis."""
        print(f"  🤔 Reflecting on: {hypothesis.title[:30]}...")
        review = {
            "score": 0.75,
            "strengths": ["Mechanistically plausible"],
            "weaknesses": ["Lack of clinical data"],
            "suggestions": ["Add a control group for variable Z"]
        }
        self.reviews_completed += 1
        return review
