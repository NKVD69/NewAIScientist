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
from typing import Any
from utils import bradley_terry as bt
from utils.budget import BudgetTracker, enable_from_env
from utils.convergence import ConvergenceTracker
from utils.pipeline import FailurePolicy, TaskSpec

# Re-exported for legacy callers (e.g. scripts/generate_paper.py).
import config  # noqa: F401
from agents import (
    AnalysisAgent,
    EvolutionAgent,
    ExperimentAgent,
    GenerationAgent,
    GraphAgent,
    LiteratureAgent,
    MetaReviewAgent,
    ProtocolAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
    ScopingAgent,
    SupervisorAgent,
    WritingAgent,
    PreregistrationAgent,
    ReplicationAgent,
)
from models import (
    AnalysisPlan,
    ContextMemory,
    Hypothesis,
    HypothesisStatus,
    ResearchGoal,
    ReviewCritique,
    StudyPhase,
    TournamentMatch,
    UserFeedback,
)

logger = logging.getLogger(__name__)

# -- Backward-compatible re-exports (DEPRECATED — use utils.llm directly) ----
# scripts/generate_paper.py and possibly other external callers still
# import these legacy names from co_scientist; keep them re-exported.
# (Each line gets its own noqa so ruff's autofix never drops them again.)
from utils.llm import ensure_str as _ensure_str  # noqa: F401, E402
from utils.llm import get_llm_completion as _get_llm_completion  # noqa: F401, E402
from utils.llm import get_llm_usage_stats  # noqa: F401, E402
from utils.llm import parse_json_response as _parse_json_response  # noqa: F401, E402
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

    def __init__(
        self,
        use_local_llm: bool = True,
        enable_rag: bool = True,
        budget: BudgetTracker | None = None,
        max_parallel: int = 4,
    ):
        self.name = "Orchestrator"
        self.context_memory = ContextMemory()
        # Budgeting: explicit tracker wins, else environment, else unlimited.
        # A run with no ceiling was the previous unconditional default.
        self.budget = budget if budget is not None else enable_from_env()
        self.supervisor = SupervisorAgent(
            max_parallel=max_parallel, budget=self.budget,
        )
        self.run_reports = []

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
        self.preregistration_agent = PreregistrationAgent(use_local_llm=use_local_llm)
        self.replication_agent = ReplicationAgent(use_local_llm=use_local_llm)

        # Register all agents with supervisor
        for agent in [
            self.generation_agent, self.reflection_agent, self.ranking_agent,
            self.proximity_agent, self.evolution_agent, self.meta_review_agent,
            self.literature_agent, self.graph_agent, self.experiment_agent,
            self.scoping_agent, self.protocol_agent, self.analysis_agent,
            self.writing_agent, self.preregistration_agent, self.replication_agent, self,
        ]:
            self.supervisor.register_agent(agent)

    # ------------------------------------------------------------------
    # RESEARCH GOAL
    # ------------------------------------------------------------------

    async def initialize_research_goal(self,
                                       title: str,
                                       description: str,
                                       domain: str,
                                       preferences: dict = None,
                                       constraints: list[str] = None) -> ResearchGoal:
        """Initialize research goal from scientist input."""
        goal = ResearchGoal(
            title=title,
            description=description,
            domain=domain,
            preferences=preferences or {},
            constraints=constraints or [],
        )
        self.context_memory.research_goal = goal
        print("\n📋 Research Goal Initialized:")
        print(f"   Title: {goal.title}")
        print(f"   Domain: {goal.domain}")
        print(f"   Description: {goal.description[:100]}...")
        return goal

    async def analyze_research_description(self, description: str) -> dict[str, Any]:
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
                                    sources: list[str] = None,
                                    iterations: int = 2) -> list[dict]:
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
            print("\n🧠 Processing papers with RAG system...")
            chunks = await self.literature_agent.process_papers_with_rag(papers)
            if chunks > 0:
                print(f"✓ RAG system ready with {chunks} indexed chunks")

        return papers

    # ------------------------------------------------------------------
    # PHASE 1: SCOPING
    # ------------------------------------------------------------------

    async def run_scoping_cycle(self) -> dict:
        """Run the research scoping phase."""
        self.context_memory.current_phase = StudyPhase.SCOPING.value
        print("\n🔍 [Scoping Phase] Analyzing research goal and literature...")

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

    async def run_hypothesis_generation_cycle(self, num_hypotheses: int = 5) -> list[Hypothesis]:
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

    async def run_review_cycle(self) -> list[ReviewCritique]:
        """Review all unreviewed hypotheses."""
        print("\n📝 Conducting hypothesis reviews...")
        unreviewed = [h for h in self.context_memory.hypotheses.values() if len(h.reviews) == 0]

        # Ground novelty in a prior-art search before reviewing. Batched so
        # the Semantic Scholar rate limiter is respected, and done up front
        # because novelty carries weight 0.25 in the ranking prior — a
        # fabricated value there propagates into every later selection.
        if unreviewed:
            try:
                from utils.novelty import apply_report, assess_many

                reports = await assess_many(
                    unreviewed,
                    rag_engine=getattr(self.literature_agent, "rag_engine", None),
                    graph_agent=getattr(self, "graph_agent", None),
                )
                assessed = 0
                for hyp in unreviewed:
                    report = reports.get(hyp.id)
                    if report is not None:
                        apply_report(hyp, report)
                        assessed += 1 if report.searched else 0
                print(f"  🔍 Prior-art search: {assessed}/{len(unreviewed)} hypotheses assessed")
                flagged = [
                    h for h in unreviewed
                    if any("prior art" in lim.lower() for lim in h.limitations)
                ]
                for hyp in flagged:
                    print(f"  ⚠ Possible prior art for '{hyp.title[:45]}'")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch novelty assessment failed: %s", exc)

        reviews = []
        for hyp in unreviewed:
            review = await self.reflection_agent.review_hypothesis(hyp, self.context_memory.research_goal)
            reviews.append(review)
            hyp.status = HypothesisStatus.REVIEWED
        print(f"✓ Completed {len(reviews)} reviews")
        return reviews

    async def run_tournament_cycle(
        self,
        num_matches: int | None = None,
        pairing: str = "bradley_terry",
        stop_when_separated: bool = True,
    ) -> list[TournamentMatch]:
        """Conduct tournament matches under a Bayesian Bradley-Terry model.

        ``num_matches=None`` (the default) sizes the round from the pool via
        ``bradley_terry.recommended_budget`` — roughly ``2·n·log₂n``. The old
        fixed default of 4 matches gave ~1.7 games per hypothesis for a pool
        of 14, where ~53 are needed for the ranking to be identified at all;
        the reported ordering was mostly noise, and it selected the hypothesis
        that got written up.

        ``pairing``:
          - ``"bradley_terry"`` (default): maximise expected information,
            weighted by belief uncertainty, so newly-evolved hypotheses with
            wide error bars get tested rather than left unplayed.
          - ``"swiss"`` / ``"information_gain"`` / ``"random"``: legacy pairers.

        With ``stop_when_separated`` the round ends early once the leader is
        more than 2σ clear of the runner-up — and, more importantly, keeps
        playing when it is not.
        """
        hyp_list = list(self.context_memory.hypotheses.values())
        if len(hyp_list) < 2:
            print("  ⚠ Need at least 2 hypotheses for tournament")
            return []

        reviewed = [h for h in hyp_list if len(h.reviews) > 0]
        pool = reviewed if len(reviewed) >= 2 else hyp_list
        by_id = {h.id: h for h in pool}

        if num_matches is None:
            num_matches = bt.recommended_budget(len(pool))

        print(
            f"\n🏆 Tournament: {len(pool)} hypotheses, budget {num_matches} matches "
            f"(pairing={pairing})..."
        )

        history_pairs = [
            (m.hypothesis_a_id, m.hypothesis_b_id)
            for m in self.context_memory.tournament_history
        ]

        if pairing == "bradley_terry":
            ratings = {h.id: self.ranking_agent.get_rating(h) for h in pool}
            pair_plan = bt.plan_matches(
                ratings, num_matches=num_matches, history=history_pairs,
            ).pairs
        elif pairing == "swiss":
            from utils.tournament_pairing import swiss_pairing
            competitors = [(h.id, h.rating_mu) for h in pool]
            pair_plan, played = [], list(history_pairs)
            while len(pair_plan) < num_matches:
                round_pairs = swiss_pairing(competitors, history=played)
                if not round_pairs:
                    break
                for p in round_pairs:
                    if len(pair_plan) >= num_matches:
                        break
                    pair_plan.append(p)
                    played.append(p)
        elif pairing == "information_gain":
            from utils.tournament_pairing import information_gain_pairing
            competitors = [(h.id, h.rating_mu) for h in pool]
            pair_plan = information_gain_pairing(
                competitors, num_matches=num_matches, history=history_pairs,
            )
        else:  # legacy random
            pair_plan = []
            for _ in range(num_matches):
                a = random.choice(pool)
                b = random.choice([h for h in pool if h.id != a.id])
                pair_plan.append((a.id, b.id))

        matches = []
        for a_id, b_id in pair_plan:
            hyp_a, hyp_b = by_id.get(a_id), by_id.get(b_id)
            if hyp_a is None or hyp_b is None:
                continue
            _, match = await self.ranking_agent.conduct_tournament_match(hyp_a, hyp_b)
            matches.append(match)
            self.context_memory.tournament_history.append(match)

            if stop_when_separated and len(matches) >= max(4, len(pool)):
                ratings = {h.id: self.ranking_agent.get_rating(h) for h in pool}
                separated, detail = bt.is_separated(ratings, top_k=1)
                if separated:
                    print(f"  ✓ Leader separated after {len(matches)} matches — {detail}")
                    break

        reliability = self.ranking_agent.judge_reliability()
        rate = reliability.get("order_invariance_rate")
        if rate is not None:
            print(f"✓ {len(matches)} matches. Judge order-invariance: {rate:.0%}")
            if rate < 0.6:
                print(
                    "  ⚠ The judge changes its mind when A and B are swapped more "
                    "often than not. It is reading position, not content — treat "
                    "this ranking as unreliable."
                )
        else:
            print(f"✓ Completed {len(matches)} tournament matches")
        return matches

    # ------------------------------------------------------------------
    # Selection helper — conservative ranking
    # ------------------------------------------------------------------

    def top_hypotheses(self, n: int, conservative: bool = True) -> list[Hypothesis]:
        """Return the top ``n`` hypotheses.

        Ranks on μ − 2σ by default, so a hypothesis that won one lucky match
        cannot displace one that survived twenty. Every downstream selection
        (evolve, experiment, replicate, protocol, write-up) goes through here,
        which is what makes the uncertainty actually load-bearing rather than
        merely reported.
        """
        key = (
            (lambda h: h.rating_conservative) if conservative
            else (lambda h: h.rating_mu)
        )
        return sorted(self.context_memory.hypotheses.values(), key=key, reverse=True)[:n]

    async def run_evolution_cycle(self) -> list[Hypothesis]:
        """Evolve top hypotheses using diverse strategies."""
        print("\n🧬 Evolving hypotheses...")
        top_hyps = self.top_hypotheses(3)

        strategies = ["enhancement", "simplification", "out_of_box"]
        evolved = []
        for hyp, strategy in zip(top_hyps, strategies, strict=False):
            new_hyp = await self.evolution_agent.evolve_hypothesis(hyp, strategy=strategy)
            self.context_memory.hypotheses[new_hyp.id] = new_hyp
            evolved.append(new_hyp)

        print(f"✓ Evolved {len(evolved)} hypotheses")
        return evolved

    async def run_experiment_cycle(self) -> list[str]:
        """Run experiments on top hypotheses."""
        print("\n🧪 Running experiments...")
        top_hyps = self.top_hypotheses(2)
        results = []
        for hyp in top_hyps:
            result = await self.experiment_agent.run_experiment(hyp, self.context_memory.research_goal)
            results.append(result)
        return results

    async def run_preregistration_cycle(self) -> list[Any]:
        """Formalize free-text predictions for all generated/evolved hypotheses."""
        print("\n📋 [Preregistration Phase] Formalizing predictions...")
        results = []
        for hyp in self.context_memory.hypotheses.values():
            if not hyp.falsifiable_predictions:
                predictions = await self.preregistration_agent.formalize_predictions(
                    hyp, self.context_memory.research_goal
                )
                results.append(predictions)
        return results

    async def run_replication_cycle(self) -> list[dict]:
        """Replicate experiments to assess reproducibility for top hypotheses."""
        print("\n🧬 [Replication Phase] Verifying reproducibility of top hypotheses...")
        top_hyps = self.top_hypotheses(2)
        results = []
        for hyp in top_hyps:
            result = await self.replication_agent.replicate_experiment(
                hyp, self.context_memory.research_goal
            )
            results.append(result)
        return results

    async def run_revision_cycle(self) -> list[Hypothesis]:
        """Revise hypotheses whose pre-registered predictions were refuted.

        Replaces the substring search that used to drive this decision::

            if "fail" in results.lower() or "reject" in results.lower():

        which fired on "failed to reject the null hypothesis" — the standard
        phrase for *absence of evidence against* — and missed unambiguous
        quantitative refutations that happened not to use those words.

        Refutation is now decided per prediction by ``utils.adjudication``,
        which compares each observed measurement to the pre-registered
        ``refuting_threshold``. Before trusting any verdict we verify the
        pre-registration hash: if the predictions moved after registration,
        the hypothesis was HARKed and its verdicts are worthless.
        """
        from models.experiment import VerdictStatus

        print("\n🔄 [Revision Phase] Adjudicating hypotheses against pre-registered predictions...")
        revised: list[Hypothesis] = []
        tampered: list[str] = []

        for hyp in list(self.context_memory.hypotheses.values()):
            if not hyp.experiment_runs:
                continue

            # --- Anti-HARKing gate ---------------------------------------
            intact, integrity_detail = self.preregistration_agent.check_integrity(hyp)
            if not intact and hyp.prediction_hash:
                print(f"  🚫 '{hyp.title[:50]}': {integrity_detail} — verdicts discarded.")
                tampered.append(hyp.id)
                hyp.limitations.append(f"Pre-registration integrity failure: {integrity_detail}")
                continue

            refuted = [
                v for v in hyp.verdicts
                if v.get("status") == VerdictStatus.REFUTED.value
            ]
            untested = [
                v for v in hyp.verdicts
                if v.get("status") in (
                    VerdictStatus.UNTESTED.value, VerdictStatus.INVALID.value,
                )
            ]

            if not refuted:
                if untested and len(untested) == len(hyp.verdicts):
                    # Nothing was actually tested. Say so, rather than letting
                    # silence read as support (the old failure mode).
                    print(
                        f"  ⬜ '{hyp.title[:50]}': {len(untested)} prediction(s) "
                        "untested — no empirical claim can be made."
                    )
                continue

            detail = "; ".join(
                f"{v['quantity']}: expected {v.get('expected')} {v.get('unit', '')}, "
                f"observed {v.get('observed')}"
                for v in refuted[:3]
            )
            print(
                f"  ❌ '{hyp.title[:50]}': {len(refuted)}/{len(hyp.verdicts)} "
                f"prediction(s) refuted — {detail}. Triggering revision..."
            )

            new_hyp = await self.evolution_agent.evolve_hypothesis(
                hyp, strategy="experimental_revision", refutations=refuted,
            )
            self.context_memory.hypotheses[new_hyp.id] = new_hyp
            revised.append(new_hyp)

        if tampered:
            print(f"⚠ {len(tampered)} hypothesis(es) failed the pre-registration integrity check.")
        print(f"✓ Revised {len(revised)} hypotheses on the basis of adjudicated refutations")
        return revised

    # ------------------------------------------------------------------
    # PHASE 3.5: INTERACTIVE FEEDBACK LOOP
    # ------------------------------------------------------------------

    async def run_interactive_feedback_cycle(
        self,
        top_n: int = 3,
        feedbacks: list[UserFeedback] | None = None,
    ) -> list[Hypothesis]:
        """Inject scientist feedback into hypothesis evolution.

        If ``feedbacks`` is provided, those entries drive the cycle (for
        non-interactive callers and tests). Otherwise the CLI driver is
        invoked over the top ``top_n`` hypotheses by Elo.

        Returns the list of newly-evolved hypotheses (may be empty).
        """
        top = self.top_hypotheses(top_n)
        if not top:
            logger.info("No hypotheses available for interactive feedback.")
            return []

        if feedbacks is None:
            from utils.interactive_feedback import collect_feedback_cli
            feedbacks = collect_feedback_cli(top)

        evolved: list[Hypothesis] = []
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
                # A scientist rejecting a hypothesis is strong evidence.
                # Move the belief itself (and tighten it: this is not noise),
                # not just the mirrored elo_rating field.
                hyp.rating_mu = max(0.0, hyp.rating_mu - 200.0)
                hyp.rating_sigma = max(bt.MIN_SIGMA, hyp.rating_sigma * 0.8)
                hyp.elo_rating = hyp.rating_mu
                self.ranking_agent.ratings[hyp.id] = bt.Rating(
                    mu=hyp.rating_mu, sigma=hyp.rating_sigma,
                    matches=hyp.rating_matches,
                )

        logger.info(
            "Interactive feedback: %d feedbacks ⇒ %d evolved hypotheses.",
            len(feedbacks), len(evolved),
        )
        return evolved

    async def run_meta_review_cycle(self) -> dict[str, Any]:
        """Meta-review: synthesize insights across all hypotheses."""
        print("\n🔄 Generating meta-review...")
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
            top = self.top_hypotheses(1)
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
    # HYPOTHESIS MUTATION (PATCH endpoint)
    # ------------------------------------------------------------------

    def update_hypothesis(self, hypothesis_id: str, updates: dict) -> Hypothesis | None:
        """Apply partial updates to a stored hypothesis.

        Only whitelisted fields can be patched, to prevent callers from
        corrupting Elo / status / parent IDs via the public API.
        Returns the mutated hypothesis, or None if the ID is unknown.
        """
        allowed = {"scientist_notes", "limitations", "testable_predictions"}
        hyp = self.context_memory.hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        for key, value in (updates or {}).items():
            if key in allowed and hasattr(hyp, key):
                setattr(hyp, key, value)
        return hyp

    # ------------------------------------------------------------------
    # PHASE 6: WRITING
    # ------------------------------------------------------------------

    async def run_writing_cycle(self) -> Any:
        """Generate the final research paper."""
        self.context_memory.current_phase = StudyPhase.WRITING.value
        print("\n📝 [Writing Phase] Drafting scientific manuscript...")
        goal = self.context_memory.research_goal

        top_hyps = self.top_hypotheses(len(self.context_memory.hypotheses) or 1)
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
        print("✓ Manuscript compiled and exported.")
        return manuscript

    # ------------------------------------------------------------------
    async def run_proximity_cycle(self):
        """Wrapper for computing proximity to be called via task queue."""
        print("\n🔗 Computing hypothesis proximity...")
        await self.proximity_agent.compute_proximity(
            list(self.context_memory.hypotheses.values()),
        )

    async def run_meta_review_and_status(self, iteration: int):
        """Wrapper for meta review and status printing to be called via task queue."""
        meta_review = await self.run_meta_review_cycle()
        self._print_iteration_status(iteration, meta_review)
        return meta_review

    # ------------------------------------------------------------------
    # FULL WORKFLOW
    # ------------------------------------------------------------------

    def _initial_pipeline(self, num_hypotheses: int = 5) -> list[TaskSpec]:
        """Phase 1: build the evidence base, then generate on top of it.

        ``literature`` is ABORT because everything downstream assumes a
        populated corpus. The old code continued past its failure, so a
        network error produced ungrounded hypotheses that nothing flagged.
        """
        return [
            TaskSpec(
                name="literature", action="run_literature_search",
                on_failure=FailurePolicy.RETRY, max_retries=2,
                description="Fetch and index papers. Everything downstream needs this.",
            ),
            TaskSpec(
                name="scoping", action="run_scoping_cycle",
                depends_on=("literature",), on_failure=FailurePolicy.DEGRADE,
                description="Narrow the goal. Useful, not load-bearing.",
            ),
            TaskSpec(
                name="generation", action="run_hypothesis_generation_cycle",
                params={"num_hypotheses": num_hypotheses},
                depends_on=("literature",), on_failure=FailurePolicy.ABORT,
                description="No hypotheses, no run.",
            ),
            TaskSpec(
                name="preregistration", action="run_preregistration_cycle",
                depends_on=("generation",), on_failure=FailurePolicy.ABORT,
                description="Seals predictions. Without it nothing can be adjudicated.",
            ),
        ]

    def _iteration_pipeline(self, iteration: int) -> list[TaskSpec]:
        """Phase 2: review, rank, evolve.

        ``review`` and ``proximity`` are independent and now run concurrently;
        the old priority integers forced them into series for no reason.
        """
        return [
            TaskSpec(name="review", action="run_review_cycle",
                     on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="proximity", action="run_proximity_cycle",
                     on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="tournament", action="run_tournament_cycle",
                     depends_on=("review",), on_failure=FailurePolicy.ABORT,
                     description="The only selection signal; a failure here "
                                 "makes every later choice arbitrary."),
            TaskSpec(name="evolution", action="run_evolution_cycle",
                     depends_on=("tournament",), on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="prereg_evolved", action="run_preregistration_cycle",
                     depends_on=("evolution",), on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="meta_review", action="run_meta_review_and_status",
                     params={"iteration": iteration},
                     depends_on=("tournament",), on_failure=FailurePolicy.DEGRADE),
        ]

    def _validation_pipeline(self) -> list[TaskSpec]:
        """Phase 3: empirical validation and closed-loop revision."""
        return [
            TaskSpec(name="experiment", action="run_experiment_cycle",
                     on_failure=FailurePolicy.DEGRADE,
                     description="May legitimately fail (no sandbox, no data)."),
            TaskSpec(name="replication", action="run_replication_cycle",
                     depends_on=("experiment",), on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="revision", action="run_revision_cycle",
                     depends_on=("experiment",), on_failure=FailurePolicy.DEGRADE,
                     description="Adjudicated refutations drive revision."),
            TaskSpec(name="review_revised", action="run_review_cycle",
                     depends_on=("revision",), on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="prereg_revised", action="run_preregistration_cycle",
                     depends_on=("revision",), on_failure=FailurePolicy.DEGRADE),
        ]

    def _output_pipeline(self) -> list[TaskSpec]:
        """Phase 4: protocol and manuscript."""
        return [
            TaskSpec(name="protocol", action="run_protocol_cycle",
                     on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="writing", action="run_writing_cycle",
                     on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="summary", action="_print_final_summary",
                     depends_on=("writing",), on_failure=FailurePolicy.IGNORE),
        ]

    async def run_full_cycle(self, num_iterations: int = 3, num_hypotheses: int = 5):
        """Run the complete workflow as a sequence of validated task DAGs."""
        print("\n" + "=" * 70)
        print("🤖 NewAI Scientist WORKFLOW STARTED (DAG orchestration)")
        print("=" * 70)

        if self.budget is not None:
            print(f"💰 Budget: {self.budget.summary()}")
        print(f"🔒 Sandbox: {self.experiment_agent.isolation_status()['strength']}")

        tracker = ConvergenceTracker(max_iterations=num_iterations)
        self.run_reports = []

        # --- Phase 1 -----------------------------------------------------
        print("\n🚀 Phase 1 — Evidence base and hypothesis generation")
        report = await self.supervisor.run_pipeline(
            self._initial_pipeline(num_hypotheses), label="initial",
        )
        self.run_reports.append(report)
        if report.aborted:
            print("\n🛑 Aborting: the evidence base could not be established.")
            print("   Continuing would produce hypotheses grounded in nothing.")
            self._print_run_health()
            return

        # --- Phase 2 -----------------------------------------------------
        for iteration in range(1, num_iterations + 1):
            print(f"\n🔄 Phase 2 — Iteration {iteration}/{num_iterations}")
            report = await self.supervisor.run_pipeline(
                self._iteration_pipeline(iteration), label=f"iteration-{iteration}",
            )
            self.run_reports.append(report)

            if report.aborted:
                print(f"\n🛑 Iteration {iteration} aborted — stopping the loop.")
                break

            convergence = tracker.update(
                self.context_memory.hypotheses,
                self.context_memory.tournament_history,
                iteration,
            )
            if tracker.should_stop(iteration):
                print(f"\n✨ Converged at iteration {iteration}:")
                for reason in convergence.reasons:
                    print(f"  • {reason}")
                break

            if self.budget is not None and self.budget.exhausted:
                print(f"\n💰 Budget exhausted after iteration {iteration}: "
                      f"{self.budget.summary()}")
                break

        # --- Phase 3 -----------------------------------------------------
        print("\n🧪 Phase 3 — Empirical validation and closed-loop revision")
        report = await self.supervisor.run_pipeline(
            self._validation_pipeline(), label="validation",
        )
        self.run_reports.append(report)

        # --- Phase 4 -----------------------------------------------------
        print("\n📝 Phase 4 — Protocol and manuscript")
        report = await self.supervisor.run_pipeline(
            self._output_pipeline(), label="output",
        )
        self.run_reports.append(report)

        self._print_run_health()

    def _print_run_health(self) -> None:
        """Report what actually completed. Printed at the end of every run.

        The point is that a partially-failed run must be visibly partial.
        Previously a run with a dead literature phase and a crashed tournament
        produced the same triumphant summary as a clean one.
        """
        print("\n" + "=" * 70)
        print("🩺 RUN HEALTH")
        print("=" * 70)

        clean = all(r.clean for r in getattr(self, "run_reports", []))
        for report in getattr(self, "run_reports", []):
            print(f"  {report.summary()}")

        digest = self.supervisor.failure_digest()
        if "No task failures" not in digest:
            print(f"\n{digest}")

        if self.budget is not None:
            print(f"\n{self.budget.render()}")

        if clean:
            print("\n✅ All tasks completed. Results rest on a complete evidence base.")
        else:
            print(
                "\n⚠ INCOMPLETE RUN — some tasks failed or were skipped. Any "
                "manuscript or protocol produced above rests on a partial "
                "evidence base and must state so."
            )

    # ------------------------------------------------------------------
    # STATUS & EXPORT
    # ------------------------------------------------------------------

    def _print_iteration_status(self, iteration: int, meta_review: dict):
        print(f"\n📊 Iteration {iteration} Summary:")
        print(f"  Total hypotheses: {meta_review['total_hypotheses']}")
        print("\n  Top hypotheses:")
        for h_info in meta_review["top_hypotheses"][:3]:
            print(f"    • {h_info['title'][:50]}...")
            print(f"      Elo: {h_info['elo_rating']:.0f} | Novelty: {h_info['novelty']}")
        if meta_review["suggested_improvements"]:
            print("\n  Suggested improvements:")
            for s in meta_review["suggested_improvements"][:2]:
                print(f"    • {s}")
        if meta_review["next_iterations_focus"]:
            print("\n  Next focus areas:")
            for f in meta_review["next_iterations_focus"][:2]:
                print(f"    • {f}")

    async def _print_final_summary(self):
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print("\n📈 System Statistics:")
        print(f"  Total hypotheses generated: {len(self.context_memory.hypotheses)}")
        print(f"  Tournament matches: {len(self.context_memory.tournament_history)}")
        print(f"  Generation agent: {self.generation_agent.generated_count} hypotheses")
        print(f"  Reflection agent: {self.reflection_agent.reviews_completed} reviews")
        print(f"  Ranking agent: {self.ranking_agent.matches_completed} matches")
        print(f"  Evolution agent: {self.evolution_agent.evolved_hypotheses} evolutions")
        print(f"  Literature agent: {self.literature_agent.papers_retrieved} papers retrieved")

        top_hyps = self.top_hypotheses(5)

        print("\n🏆 Top 5 Hypotheses (conservative rating, μ − 2σ):")
        for i, hyp in enumerate(top_hyps, 1):
            print(f"\n{i}. {hyp.title}")
            print(f"   ID: {hyp.id}")
            print(
                f"   Rating: μ={hyp.rating_mu:.0f} ± σ={hyp.rating_sigma:.0f} "
                f"(conservative {hyp.rating_conservative:.0f}, "
                f"{hyp.rating_matches} matches)"
            )
            if hyp.verdicts:
                statuses = [v.get("status", "?") for v in hyp.verdicts]
                print(f"   Adjudication: {', '.join(statuses)}")
                print(f"   Empirical support: {hyp.empirical_support:+.2f}")
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
                    "rating_mu": h.rating_mu,
                    "rating_sigma": h.rating_sigma,
                    "rating_conservative": h.rating_conservative,
                    "rating_matches": h.rating_matches,
                    "empirical_support": h.empirical_support,
                    "verdicts": h.verdicts,
                    "experiment_runs": h.experiment_runs,
                    "prediction_hash": h.prediction_hash,
                    "registered_at": h.registered_at,
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
