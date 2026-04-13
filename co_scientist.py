import os
import json
import asyncio
import uuid
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

from models import (
    ResearchGoal, 
    Hypothesis, 
    ContextMemory, 
    StudyPhase, 
    ExperimentalProtocol,
    StatisticalResult,
    Manuscript
)

from agents import (
    ScopingAgent,
    ProtocolAgent,
    AnalysisAgent,
    WritingAgent,
    DevilsAdvocateAgent,
    HypothesisChainingAgent,
    RankingAgent,
    EvolutionAgent,
    GenerationAgent,
    ReflectionAgent,
    SearchAgent,
    GraphAgent,
    MetaReviewAgent
)

class CoScientist:
    """
    Main orchestrator for the NewAI Scientist v3.0 platform.
    Coordinates specialized agents through a 6-phase scientific workflow.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.context_memory = ContextMemory()
        
        # Initialize Core Agents
        self.scoping_agent = ScopingAgent(model_name)
        self.protocol_agent = ProtocolAgent(model_name)
        self.analysis_agent = AnalysisAgent(model_name)
        self.writing_agent = WritingAgent(model_name)
        
        # Initialize specialized v3.0 Agents
        self.critic_agent = DevilsAdvocateAgent(model_name)
        self.chaining_agent = HypothesisChainingAgent(model_name)
        self.ranking_agent = RankingAgent(model_name)
        self.evolution_agent = EvolutionAgent(model_name)
        self.generation_agent = GenerationAgent(model_name)
        self.reflection_agent = ReflectionAgent(model_name)
        
        # Utility Agents
        self.search_agent = SearchAgent(model_name)
        self.graph_agent = GraphAgent(model_name)
        self.meta_review_agent = MetaReviewAgent(model_name)

    async def initialize_research_goal(
        self, 
        title: str, 
        description: str, 
        domain: str = "Biomedicine",
        preferences: Dict = None,
        constraints: List[str] = None
    ) -> ResearchGoal:
        """Sets the initial research mission."""
        goal = ResearchGoal(
            title=title,
            description=description,
            domain=domain,
            preferences=preferences or {},
            constraints=constraints or []
        )
        self.context_memory.research_goal = goal
        self.context_memory.current_phase = StudyPhase.SCOPING
        print(f"🚀 Mission Initialized: {title}")
        return goal

    # --------------------------------------------------------------------------
    # Phase 1-6 Workflow Methods
    # --------------------------------------------------------------------------

    async def run_scoping_cycle(self) -> Dict:
        """Phase 1: Research Scoping and Gap Analysis."""
        self.context_memory.current_phase = StudyPhase.SCOPING
        result = await self.scoping_agent.analyze_gaps(self.context_memory.research_goal, [])
        self.context_memory.conceptual_framework = result.get("framework")
        return result

    async def run_literature_search(self, query: str = None, max_results: int = 5) -> List[Dict]:
        """Phase 2: Literature Retrieval and Context Building."""
        self.context_memory.current_phase = StudyPhase.LITERATURE_REVIEW
        search_query = query or self.context_memory.research_goal.title
        papers = await self.search_agent.search(search_query, max_results)
        self.context_memory.literature_context.extend(papers)
        return papers

    async def run_hypothesis_generation_cycle(self, num_hypotheses: int = 5) -> List[Hypothesis]:
        """Phase 3: Generation, Adversarial Review, and Chaining."""
        self.context_memory.current_phase = StudyPhase.HYPOTHESIS_GENERATION
        
        # 1. Generation
        new_hyps = await self.generation_agent.generate_batch(
            self.context_memory.research_goal, 
            self.context_memory.literature_context,
            count=num_hypotheses
        )
        
        for hyp in new_hyps:
            # 2. Reflection & Adversarial Critique (Sprint 3)
            review = await self.reflection_agent.review(hyp, self.context_memory.research_goal)
            hyp.reviews.append(review)
            
            adv_critique = await self.critic_agent.refute_hypothesis(hyp, self.context_memory.research_goal, [])
            hyp.adversarial_review = adv_critique
            
            self.context_memory.hypotheses[hyp.id] = hyp
            
        # 3. Chaining Identification (Sprint 3)
        links = await self.chaining_agent.identify_links(
            list(self.context_memory.hypotheses.values()),
            self.context_memory.research_goal
        )
        # Update hygiene: apply links to models
        for link in links:
            if link["target"] in self.context_memory.hypotheses:
                self.context_memory.hypotheses[link["target"]].parent_id = link["source"]
                self.context_memory.hypotheses[link["target"]].link_type = link["type"]

        return list(self.context_memory.hypotheses.values())

    async def run_protocol_cycle(self, hypothesis_id: str = None) -> ExperimentalProtocol:
        """Phase 4: Experimental Design and Power Analysis."""
        self.context_memory.current_phase = StudyPhase.EXPERIMENTAL_DESIGN
        hyp = self._get_target_hypothesis(hypothesis_id)
        protocol = await self.protocol_agent.design_protocol(hyp, self.context_memory.research_goal)
        # In a real app, store this in memory/DB
        return protocol

    async def run_analysis_cycle(self, hypothesis_id: str = None, file_path: str = None) -> StatisticalResult:
        """Phase 5: Data Analysis and Interpretation."""
        self.context_memory.current_phase = StudyPhase.DATA_ANALYSIS
        hyp = self._get_target_hypothesis(hypothesis_id)
        result = await self.analysis_agent.run_statistical_tests(hyp, file_path)
        return result

    async def run_writing_cycle(self) -> Manuscript:
        """Phase 6: Manuscript Compilation & Export."""
        self.context_memory.current_phase = StudyPhase.WRITING
        manuscript = await self.writing_agent.draft_manuscript(
            self.context_memory.research_goal,
            list(self.context_memory.hypotheses.values()),
            {} # Analysis results
        )
        return manuscript

    async def update_hypothesis(self, hyp_id: str, new_data: Dict) -> Hypothesis:
        """Updates a hypothesis and saves the current state to history."""
        if hyp_id not in self.context_memory.hypotheses:
            raise ValueError("Hypothesis not found")
        
        hyp = self.context_memory.hypotheses[hyp_id]
        # Save snapshot to history
        snapshot = {
            "version": len(hyp.history) + 1,
            "timestamp": datetime.now().isoformat(),
            "title": hyp.title,
            "description": hyp.description,
            "status": hyp.status.value
        }
        hyp.history.append(snapshot)
        
        # Apply updates
        for key, value in new_data.items():
            if hasattr(hyp, key):
                setattr(hyp, key, value)
        
        return hyp

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    def _get_target_hypothesis(self, hyp_id: str = None) -> Hypothesis:
        if hyp_id and hyp_id in self.context_memory.hypotheses:
            return self.context_memory.hypotheses[hyp_id]
        if self.context_memory.hypotheses:
            # Return top rated if no ID provided
            return sorted(self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True)[0]
        raise ValueError("No hypotheses available for this phase.")

    async def run_full_cycle(self, num_iterations: int = 1):
        """Executes the complete scientific research pipeline."""
        print(f"\n{'='*70}\nSTARTING FULL SCIENTIFIC CYCLE\n{'='*70}")
        
        # 1. Scoping
        await self.run_scoping_cycle()
        
        # 2. Literature
        await self.run_literature_search(max_results=10)
        
        # 3. Iterative Hypothesis refinement
        for i in range(num_iterations):
            print(f"\n🔄 Iteration {i+1}/{num_iterations}")
            await self.run_hypothesis_generation_cycle(num_hypotheses=3)
            await self.graph_agent.update_graph(self.context_memory)
            
        # 4. Protocol for top hypothesis
        await self.run_protocol_cycle()
        
        # 5. Writing
        await self.run_writing_cycle()
        
        print(f"\n{'='*70}\nWORKFLOW COMPLETE\n{'='*70}")

# ============================================================================
# MAIN EXECUTION (Local Test)
# ============================================================================

if __name__ == "__main__":
    async def test():
        sci = CoScientist()
        await sci.initialize_research_goal(
            title="SGLT2 inhibitors for Neuroprotection",
            description="Explore the repurposing of diabetes drugs for Alzheimer's prevention.",
            domain="Biomedicine"
        )
        await sci.run_full_cycle()

    asyncio.run(test())
