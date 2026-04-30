"""
NewAI Scientist v3.0 — Orchestrator

Lightweight coordinator that delegates all work to specialized agents
in the `agents/` package. Manages the 6-phase scientific workflow:

  1. Literature → 2. Scoping → 3. Hypotheses (Generation + Review + Tournament + Evolution)
     → 4. Experimental Design → 5. Analysis → 6. Writing

All agent implementations live in agents/*.py.
All data models live in models/*.py.
LLM utilities live in utils/llm.py.
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import config
from models import (
    HypothesisStatus,
    StudyPhase,
    ReviewCritique,
    Hypothesis,
    ResearchGoal,
    ResearchQuestion,
    TournamentMatch,
    ContextMemory,
    AnalysisPlan,
    UserFeedback,
)
from agents import (
    LiteratureAgent,
    GenerationAgent,
    ReflectionAgent,
    RankingAgent,
    ProximityAgent,
    EvolutionAgent,
    MetaReviewAgent,
    GraphAgent,
    ExperimentAgent,
    SupervisorAgent,
    ScopingAgent,
    ProtocolAgent,
    AnalysisAgent,
    WritingAgent,
)

logger = logging.getLogger(__name__)

# -- Backward-compatible re-exports (DEPRECATED — use utils.llm directly) ----
from utils.llm import (                     # noqa: F401, E402
    get_llm_completion as _get_llm_completion,
    parse_json_response as _parse_json_response,
    ensure_str as _ensure_str,
    get_llm_usage_stats,
)
from utils.safety import check_code_safety as _check_code_safety  # noqa: F401, E402

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================================
# MAIN CO-SCIENTIST SYSTEM
# ============================================================================

class CoScientist:
    """Main AI co-scientist system coordinator — pure orchestration, no agent logic."""

    def __init__(self, use_local_llm: bool = True, enable_rag: bool = True):
        self.context_memory = ContextMemory()
        self.supervisor = SupervisorAgent()

        # Core agents (v2.2)
        self.generation_agent = GenerationAgent(use_local_llm=use_local_llm)
        self.reflection_agent = ReflectionAgent(use_local_llm=use_local_llm)
        self.ranking_agent = RankingAgent(use_local_llm=use_local_llm)
        self.proximity_agent = ProximityAgent()
        self.evolution_agent = EvolutionAgent(use_local_llm=use_local_llm)
        self.meta_review_agent = MetaReviewAgent()
        self.literature_agent = LiteratureAgent(use_local_llm=use_local_llm, enable_rag=enable_rag)
        self.graph_agent = GraphAgent(use_local_llm=use_local_llm)
        self.experiment_agent = ExperimentAgent(use_local_llm=use_local_llm)

        # v3.0 New Agents
        self.scoping_agent = ScopingAgent(use_local_llm=use_local_llm)
        self.protocol_agent = ProtocolAgent(use_local_llm=use_local_llm)
        self.analysis_agent = AnalysisAgent(use_local_llm=use_local_llm)
        self.writing_agent = WritingAgent(use_local_llm=use_local_llm)

        # Register all agents with supervisor
        for agent in [
            self.generation_agent, self.reflection_agent, self.ranking_agent,
            self.proximity_agent, self.evolution_agent, self.meta_review_agent,
            self.literature_agent, self.graph_agent, self.experiment_agent,
            self.scoping_agent, self.protocol_agent, self.analysis_agent,
            self.writing_agent,
        ]:
            self.supervisor.register_agent(agent)

    # ------------------------------------------------------------------
    # RESEARCH GOAL
    # ------------------------------------------------------------------

    async def initialize_research_goal(self,
                                       title: str,
                                       description: str,
                                       domain: str,
                                       preferences: Dict = None,
                                       constraints: List[str] = None) -> ResearchGoal:
        """Initialize research goal from scientist input."""
        goal = ResearchGoal(
            title=title,
            description=description,
            domain=domain,
            preferences=preferences or {},
            constraints=constraints or [],
        )
        self.context_memory.research_goal = goal
        print(f"\n📋 Research Goal Initialized:")
        print(f"   Title: {goal.title}")
        print(f"   Domain: {goal.domain}")
        print(f"   Description: {goal.description[:100]}...")
        return goal

    async def analyze_research_description(self, description: str) -> Dict[str, Any]:
        """Analyze research description to suggest domain and databases."""
        from utils.llm import get_llm_completion, parse_json_response
        if not self.generation_agent.llm_client:
            return {"domains": ["Biomedicine", "Computer Science", "Physics"], "databases": ["arxiv", "pubmed"]}

        prompt = f"""Analyze the following research description and identify:
1. The most relevant scientific domains.
2. The most relevant scientific databases for literature search.

Description: "{description}"

Provide the output in JSON format with two keys: 'domains' (list of strings) and 'databases' (list of strings).
Supported databases: ['arxiv', 'pubmed', 'biorxiv', 'ieee_xplore', 'scopus']."""

        try:
            response = await get_llm_completion(
                self.generation_agent.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, json_mode=True,
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠ Domain analysis failed: {e}")
            return {"domains": ["Science", "Research"], "databases": ["arxiv", "pubmed"]}

    # ------------------------------------------------------------------
    # PHASE 2: LITERATURE
    # ------------------------------------------------------------------

    async def run_literature_search(self, max_results: int = 5,
                                    sources: List[str] = None,
                                    iterations: int = 2) -> List[Dict]:
        """Run literature search to populate context."""
        if sources is None:
            sources = ["arxiv"]
        self.context_memory.current_phase = StudyPhase.LITERATURE_REVIEW.value
        print(f"\n📚 Running literature search on {sources} (Max {iterations} iterations)...")

        papers = await self.literature_agent.search_literature(
            self.context_memory.research_goal,
            max_results=max_results, sources=sources, iterations=iterations,
        )
        self.context_memory.literature_context = papers

        # CAG + GraphRAG
        cag_context = await self.literature_agent.extract_key_findings(papers, self.context_memory.research_goal)
        graph_insights = await self.graph_agent.build_graph(papers, self.context_memory.research_goal)
        cag_context += "\n\n" + graph_insights
        print(f"🧠 CAG + Graph Context Generated: {len(cag_context)} chars")
        self.generation_agent.cag_context = cag_context

        # RAG indexing
        if self.literature_agent.rag_engine and papers:
            print(f"\n🧠 Processing papers with RAG system...")
            chunks = await self.literature_agent.process_papers_with_rag(papers)
            if chunks > 0:
                print(f"✓ RAG system ready with {chunks} indexed chunks")

        return papers

    # ------------------------------------------------------------------
    # PHASE 1: SCOPING
    # ------------------------------------------------------------------

    async def run_scoping_cycle(self) -> Dict:
        """Run the research scoping phase."""
        self.context_memory.current_phase = StudyPhase.SCOPING.value
        print(f"\n🔍 [Scoping Phase] Analyzing research goal and literature...")

        soa = await self.scoping_agent.analyze_state_of_art(
            self.context_memory.literature_context, self.context_memory.research_goal,
        )
        self.context_memory.state_of_art = asdict(soa)

        questions = await self.scoping_agent.generate_research_questions(
            soa, self.context_memory.research_goal,
        )
        self.context_memory.research_questions = questions

        framework = await self.scoping_agent.build_conceptual_framework(
            questions, self.context_memory.research_goal,
        )
        self.context_memory.conceptual_framework = framework

        print(f"✓ Scoping completed: {len(questions)} research questions generated.")
        return {"soa": soa, "questions": questions, "framework": framework}

    # ------------------------------------------------------------------
    # PHASE 3: HYPOTHESIS GENERATION, REVIEW, TOURNAMENT, EVOLUTION
    # ------------------------------------------------------------------

    async def run_hypothesis_generation_cycle(self, num_hypotheses: int = 5) -> List[Hypothesis]:
        """Generate initial hypotheses."""
        self.context_memory.current_phase = StudyPhase.HYPOTHESIS_GENERATION.value
        print(f"\n🔬 Generating {num_hypotheses} initial hypotheses...")

        rag_context = None
        if self.literature_agent.rag_engine:
            goal_q = f"{self.context_memory.research_goal.title} {self.context_memory.research_goal.description}"
            rag_context = await self.literature_agent.query_rag(goal_q, top_k=5)
            if rag_context:
                print(f"  ✓ Retrieved {len(rag_context)} relevant passages from papers")

        hypotheses = await self.generation_agent.generate_initial_hypotheses(
            self.context_memory.research_goal,
            context_papers=self.context_memory.literature_context,
            rag_context=rag_context, count=num_hypotheses,
        )
        for h in hypotheses:
            self.context_memory.hypotheses[h.id] = h
        print(f"✓ Generated {len(hypotheses)} hypotheses")
        return hypotheses

    async def run_review_cycle(self) -> List[ReviewCritique]:
        """Review all unreviewed hypotheses."""
        print(f"\n📝 Conducting hypothesis reviews...")
        unreviewed = [h for h in self.context_memory.hypotheses.values() if len(h.reviews) == 0]
        reviews = []
        for hyp in unreviewed:
            review = await self.reflection_agent.review_hypothesis(hyp, self.context_memory.research_goal)
            reviews.append(review)
            hyp.status = HypothesisStatus.REVIEWED
        print(f"✓ Completed {len(reviews)} reviews")
        return reviews

    async def run_tournament_cycle(self, num_matches: int = 5) -> List[TournamentMatch]:
        """Conduct tournament matches."""
        print(f"\n🏆 Running tournament matches...")
        hyp_list = list(self.context_memory.hypotheses.values())
        if len(hyp_list) < 2:
            print("  ⚠ Need at least 2 hypotheses for tournament")
            return []

        reviewed = [h for h in hyp_list if len(h.reviews) > 0]
        matches = []
        for _ in range(min(num_matches, len(reviewed) * 2)):
            pool = reviewed if len(reviewed) >= 2 else hyp_list
            if len(pool) < 2:
                break
            hyp_a = random.choice(pool)
            hyp_b = random.choice([h for h in pool if h.id != hyp_a.id])
            winner_id, match = await self.ranking_agent.conduct_tournament_match(hyp_a, hyp_b)
            matches.append(match)
            self.context_memory.tournament_history.append(match)

        print(f"✓ Completed {len(matches)} tournament matches")
        return matches

    async def run_evolution_cycle(self) -> List[Hypothesis]:
        """Evolve top hypotheses using diverse strategies."""
        print(f"\n🧬 Evolving hypotheses...")
        top_hyps = sorted(
            self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True,
        )[:3]

        strategies = ["enhancement", "simplification", "out_of_box"]
        evolved = []
        for hyp, strategy in zip(top_hyps, strategies):
            new_hyp = await self.evolution_agent.evolve_hypothesis(hyp, strategy=strategy)
            self.context_memory.hypotheses[new_hyp.id] = new_hyp
            evolved.append(new_hyp)

        print(f"✓ Evolved {len(evolved)} hypotheses")
        return evolved

    async def run_experiment_cycle(self) -> List[str]:
        """Run experiments on top hypotheses."""
        print(f"\n🧪 Running experiments...")
        top_hyps = sorted(
            self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True,
        )[:2]
        results = []
        for hyp in top_hyps:
            result = await self.experiment_agent.run_experiment(hyp, self.context_memory.research_goal)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # PHASE 3.5: INTERACTIVE FEEDBACK LOOP
    # ------------------------------------------------------------------

    async def run_interactive_feedback_cycle(
        self,
        top_n: int = 3,
        feedbacks: Optional[List[UserFeedback]] = None,
    ) -> List[Hypothesis]:
        """Inject scientist feedback into hypothesis evolution.

        If ``feedbacks`` is provided, those entries drive the cycle (for
        non-interactive callers and tests). Otherwise the CLI driver is
        invoked over the top ``top_n`` hypotheses by Elo.

        Returns the list of newly-evolved hypotheses (may be empty).
        """
        top = sorted(
            self.context_memory.hypotheses.values(),
            key=lambda h: h.elo_rating,
            reverse=True,
        )[:top_n]
        if not top:
            logger.info("No hypotheses available for interactive feedback.")
            return []

        if feedbacks is None:
            from utils.interactive_feedback import collect_feedback_cli
            feedbacks = collect_feedback_cli(top)

        evolved: List[Hypothesis] = []
        by_id = {h.id: h for h in top}
        for fb in feedbacks:
            hyp = by_id.get(fb.hypothesis_id)
            if hyp is None:
                logger.warning("Feedback for unknown hypothesis %s — ignored.", fb.hypothesis_id)
                continue
            new_hyp = await self.evolution_agent.evolve_with_feedback(hyp, fb)
            if new_hyp is not None:
                self.context_memory.hypotheses[new_hyp.id] = new_hyp
                evolved.append(new_hyp)
            elif fb.verdict == "disagree":
                # Mark the rejected hypothesis so downstream cycles deprioritise it.
                hyp.elo_rating = max(0.0, hyp.elo_rating - 200.0)

        logger.info(
            "Interactive feedback: %d feedbacks ⇒ %d evolved hypotheses.",
            len(feedbacks), len(evolved),
        )
        return evolved

    async def run_meta_review_cycle(self) -> Dict[str, Any]:
        """Meta-review: synthesize insights across all hypotheses."""
        print(f"\n🔄 Generating meta-review...")
        meta_review = await self.meta_review_agent.generate_meta_review(
            list(self.context_memory.hypotheses.values()),
            self.context_memory.tournament_history,
            self.context_memory.research_goal,
        )
        self.context_memory.meta_reviews.append(meta_review)
        return meta_review

    # ------------------------------------------------------------------
    # PHASE 4: EXPERIMENTAL DESIGN
    # ------------------------------------------------------------------

    async def run_protocol_cycle(self, hypothesis_id: str = None) -> Any:
        """Design experimental protocol for a hypothesis."""
        self.context_memory.current_phase = StudyPhase.EXPERIMENTAL_DESIGN.value
        if not hypothesis_id:
            top = sorted(self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True)
            if not top:
                return None
            hyp = top[0]
        else:
            hyp = self.context_memory.hypotheses.get(hypothesis_id)

        print(f"\n🧪 [Protocol Phase] Designing experiment for: {hyp.title}")
        protocol = await self.protocol_agent.design_experiment(hyp, self.context_memory.research_goal)
        await self.protocol_agent.power_analysis(protocol)
        await self.protocol_agent.generate_executable_code(protocol)
        self.context_memory.experimental_protocols[hyp.id] = protocol
        print(f"✓ Protocol generated for {hyp.id}")
        return protocol

    # ------------------------------------------------------------------
    # PHASE 5: ANALYSIS
    # ------------------------------------------------------------------

    async def run_analysis_cycle(self, hypothesis_id: str, file_path: str = None) -> Any:
        """Run statistical analysis on a dataset for a hypothesis."""
        self.context_memory.current_phase = StudyPhase.DATA_ANALYSIS.value
        hyp = self.context_memory.hypotheses.get(hypothesis_id)
        if not hyp:
            return None

        pd_df = None
        if file_path and os.path.exists(file_path) and pd is not None:
            dataset_info = await self.analysis_agent.load_csv(file_path)
            self.context_memory.datasets[dataset_info.id] = dataset_info
            pd_df = pd.read_csv(file_path)

        print(f"\n📊 [Analysis Phase] Analyzing data for: {hyp.title}")
        plan = AnalysisPlan(primary_analysis=f"Test {hyp.title} effects")

        if pd_df is not None:
            results = await self.analysis_agent.run_statistical_tests(pd_df, plan)
            interpretation = await self.analysis_agent.interpret_results(results, hyp)
            self.context_memory.statistical_results.extend(results)
            self.context_memory.interpretations[hyp.id] = interpretation
        else:
            print("   ⚠️ No dataset provided for analysis.")
            results, interpretation = [], "Analysis skipped: No dataset."

        return {"results": results, "interpretation": interpretation}

    # ------------------------------------------------------------------
    # PHASE 6: WRITING
    # ------------------------------------------------------------------

    async def run_writing_cycle(self) -> Any:
        """Generate the final research paper."""
        self.context_memory.current_phase = StudyPhase.WRITING.value
        print(f"\n📝 [Writing Phase] Drafting scientific manuscript...")
        goal = self.context_memory.research_goal

        top_hyps = sorted(self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True)
        best_hyp = top_hyps[0] if top_hyps else None

        sections = {}
        for stype in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
            context = {
                "results": self.context_memory.statistical_results,
                "literature": self.context_memory.literature_context[:3],
            }
            sec = await self.writing_agent.draft_section(stype, goal, best_hyp, context)
            sections[stype] = sec

        manuscript = await self.writing_agent.compile_manuscript(
            goal, sections, self.context_memory.literature_context[:10],
        )
        self.context_memory.manuscript = manuscript

        self.writing_agent.export_to_latex(manuscript, f"research_paper_{goal.id}.tex")
        self.writing_agent.export_to_docx(manuscript, f"research_paper_{goal.id}.docx")
        print(f"✓ Manuscript compiled and exported.")
        return manuscript

    # ------------------------------------------------------------------
    # FULL WORKFLOW
    # ------------------------------------------------------------------

    async def run_full_cycle(self, num_iterations: int = 3):
        """Run complete co-scientist workflow (v3.0 Extended)."""
        print("\n" + "=" * 70)
        print("🤖 NewAI Scientist v3.0 WORKFLOW STARTED")
        print("=" * 70)

        # Phase 2 → Literature
        await self.run_literature_search()
        # Phase 1 → Scoping
        await self.run_scoping_cycle()
        # Phase 3 → Hypotheses
        await self.run_hypothesis_generation_cycle(num_hypotheses=5)

        for iteration in range(num_iterations):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'=' * 70}")

            await self.run_review_cycle()

            print(f"\n🔗 Computing hypothesis proximity...")
            await self.proximity_agent.compute_proximity(
                list(self.context_memory.hypotheses.values()),
            )

            await self.run_tournament_cycle(num_matches=4)
            await self.run_evolution_cycle()
            meta_review = await self.run_meta_review_cycle()
            self._print_iteration_status(iteration + 1, meta_review)

        # Phase 4 → Protocol
        await self.run_protocol_cycle()
        # Phase 6 → Writing
        await self.run_writing_cycle()

        await self._print_final_summary()

    # ------------------------------------------------------------------
    # STATUS & EXPORT
    # ------------------------------------------------------------------

    def _print_iteration_status(self, iteration: int, meta_review: Dict):
        print(f"\n📊 Iteration {iteration} Summary:")
        print(f"  Total hypotheses: {meta_review['total_hypotheses']}")
        print(f"\n  Top hypotheses:")
        for h_info in meta_review["top_hypotheses"][:3]:
            print(f"    • {h_info['title'][:50]}...")
            print(f"      Elo: {h_info['elo_rating']:.0f} | Novelty: {h_info['novelty']}")
        if meta_review["suggested_improvements"]:
            print(f"\n  Suggested improvements:")
            for s in meta_review["suggested_improvements"][:2]:
                print(f"    • {s}")
        if meta_review["next_iterations_focus"]:
            print(f"\n  Next focus areas:")
            for f in meta_review["next_iterations_focus"][:2]:
                print(f"    • {f}")

    async def _print_final_summary(self):
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print(f"\n📈 System Statistics:")
        print(f"  Total hypotheses generated: {len(self.context_memory.hypotheses)}")
        print(f"  Tournament matches: {len(self.context_memory.tournament_history)}")
        print(f"  Generation agent: {self.generation_agent.generated_count} hypotheses")
        print(f"  Reflection agent: {self.reflection_agent.reviews_completed} reviews")
        print(f"  Ranking agent: {self.ranking_agent.matches_completed} matches")
        print(f"  Evolution agent: {self.evolution_agent.evolved_hypotheses} evolutions")
        print(f"  Literature agent: {self.literature_agent.papers_retrieved} papers retrieved")

        top_hyps = sorted(
            self.context_memory.hypotheses.values(), key=lambda h: h.elo_rating, reverse=True,
        )[:5]

        print(f"\n🏆 Top 5 Hypotheses (by Elo rating):")
        for i, hyp in enumerate(top_hyps, 1):
            print(f"\n{i}. {hyp.title}")
            print(f"   ID: {hyp.id}")
            print(f"   Elo Rating: {hyp.elo_rating:.0f}")
            print(f"   Novelty Level: {hyp.novelty_level}")
            print(f"   Status: {hyp.status.value}")
            print(f"   Reviews: {len(hyp.reviews)}")
            print(f"   Generation Method: {hyp.generation_method}")
            if hyp.testable_predictions:
                print(f"   Testable Predictions: {len(hyp.testable_predictions)}")

    def export_hypotheses_json(self, filename: str = "hypotheses.json"):
        """Export hypotheses to JSON."""
        data = {
            "research_goal": asdict(self.context_memory.research_goal),
            "literature_context": self.context_memory.literature_context,
            "hypotheses": [
                {
                    "id": h.id,
                    "title": h.title,
                    "description": h.description,
                    "mechanism": h.mechanism,
                    "elo_rating": h.elo_rating,
                    "novelty_level": h.novelty_level,
                    "status": h.status.value,
                    "testable_predictions": h.testable_predictions,
                    "grounding_evidence": h.grounding_evidence,
                    "experimental_results": getattr(h, "experimental_results", ""),
                    "generation_method": h.generation_method,
                    "num_reviews": len(h.reviews),
                    "cited_papers": h.cited_papers,
                }
                for h in self.context_memory.hypotheses.values()
            ],
            "tournament_matches": len(self.context_memory.tournament_history),
            "statistics": {
                "generation_agent": self.generation_agent.generated_count,
                "reflection_agent": self.reflection_agent.reviews_completed,
                "ranking_agent": self.ranking_agent.matches_completed,
                "evolution_agent": self.evolution_agent.evolved_hypotheses,
                "meta_review_agent": self.meta_review_agent.meta_reviews_generated,
                "literature_agent": self.literature_agent.papers_retrieved,
                "experiment_agent": self.experiment_agent.experiments_run,
            },
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Exported hypotheses to {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description="NewAI Scientist v3.0")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause after hypothesis generation to collect scientist feedback "
             "(verdict + free-text refinement) before proceeding to writing.",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of review/tournament/evolution iterations.",
    )
    args, _ = parser.parse_known_args()

    co_scientist = CoScientist()

    await co_scientist.initialize_research_goal(
        title="Novel Drug Repurposing for Acute Myeloid Leukemia",
        description=(
            "Identify FDA-approved drugs that could be repurposed for acute myeloid leukemia (AML) "
            "treatment. Focus on drugs that can inhibit leukemic cell proliferation at clinically "
            "applicable concentrations, particularly targeting MOLM-13 cell lines."
        ),
        domain="Biomedicine/Oncology",
        preferences={"focus_on_novelty": True, "require_testability": True, "prioritize_clinical_relevance": True},
        constraints=[
            "Only consider FDA-approved drugs",
            "Must have mechanism of action documentation",
            "Focus on inhibiting AML cell proliferation",
        ],
    )

    await co_scientist.run_full_cycle(num_iterations=args.iterations)

    if args.interactive:
        print("\n" + "=" * 70)
        print("🧑‍🔬 INTERACTIVE FEEDBACK LOOP")
        print("=" * 70)
        await co_scientist.run_interactive_feedback_cycle(top_n=3)

    co_scientist.export_hypotheses_json("co_scientist_results.json")

    print("\n" + "=" * 70)
    print("✨ AI Co-Scientist WORKFLOW COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
