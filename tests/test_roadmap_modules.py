"""
Tests for the roadmap modules: DAG orchestration, budgeting, multiverse
replication, IMRaD tagging and literature hygiene.

Regression-shaped: each class targets a specific defect from the audit, so a
refactor that reintroduces it fails loudly.
"""

from __future__ import annotations

import pytest

from utils import imrad
from utils.budget import (
    BudgetExhausted,
    BudgetLimits,
    BudgetTracker,
    estimate_tokens,
    price_for,
)
from utils.literature_hygiene import (
    apply_hygiene,
    canonical_id,
    deduplicate,
    extract_doi,
    normalise_title,
    quality_weight,
    titles_are_near_duplicates,
)
from utils.multiverse import (
    MultiverseReport,
    SpecificationResult,
    build_specification_code,
    enumerate_specifications,
    parse_specification_result,
)
from utils.pipeline import (
    FailurePolicy,
    Pipeline,
    PipelineError,
    TaskSpec,
    TaskState,
    topological_waves,
    validate,
)


class _Recorder:
    """Agent double recording which actions ran, with configurable failures."""

    name = "Rec"

    def __init__(self, fail: set[str] | None = None, fail_times: dict | None = None):
        self.calls: list[str] = []
        self.fail = fail or set()
        self.fail_times = dict(fail_times or {})

    async def _act(self, tag: str):
        self.calls.append(tag)
        if self.fail_times.get(tag, 0) > 0:
            self.fail_times[tag] -= 1
            raise RuntimeError(f"transient failure in {tag}")
        if tag in self.fail:
            raise RuntimeError(f"boom in {tag}")
        return f"result:{tag}"

    async def a(self): return await self._act("a")
    async def b(self): return await self._act("b")
    async def c(self): return await self._act("c")
    async def d(self): return await self._act("d")

    def sync_action(self):
        self.calls.append("sync")
        return "sync-result"


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------

class TestGraphValidation:
    def test_cycle_is_rejected(self):
        with pytest.raises(PipelineError, match="cycle"):
            validate([
                TaskSpec(name="x", action="a", depends_on=("y",)),
                TaskSpec(name="y", action="b", depends_on=("x",)),
            ])

    def test_unknown_dependency_is_rejected(self):
        with pytest.raises(PipelineError, match="unknown"):
            validate([TaskSpec(name="x", action="a", depends_on=("ghost",))])

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(PipelineError, match="duplicate"):
            validate([TaskSpec(name="x", action="a"), TaskSpec(name="x", action="b")])

    def test_independent_tasks_share_a_wave(self):
        waves = topological_waves([
            TaskSpec(name="root", action="a"),
            TaskSpec(name="left", action="b", depends_on=("root",)),
            TaskSpec(name="right", action="c", depends_on=("root",)),
            TaskSpec(name="join", action="d", depends_on=("left", "right")),
        ])
        assert waves == [["root"], ["left", "right"], ["join"]]


# ---------------------------------------------------------------------------
# The defect: failures were swallowed and downstream ran on corrupt state
# ---------------------------------------------------------------------------

class TestFailurePropagation:
    @pytest.mark.asyncio
    async def test_dependents_are_skipped_when_a_dependency_fails(self):
        """The literature→generation case: no corpus meant ungrounded output."""
        agent = _Recorder(fail={"a"})
        report = await Pipeline([
            TaskSpec(name="lit", action="a", agent="Rec", on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="gen", action="b", agent="Rec", depends_on=("lit",)),
        ], {"Rec": agent}).run()

        assert report.results["lit"].state is TaskState.FAILED
        assert report.results["gen"].state is TaskState.SKIPPED
        assert "gen" not in agent.calls, "must not run on a missing dependency"
        assert not report.clean

    @pytest.mark.asyncio
    async def test_abort_stops_the_whole_pipeline(self):
        agent = _Recorder(fail={"a"})
        report = await Pipeline([
            TaskSpec(name="lit", action="a", agent="Rec", on_failure=FailurePolicy.ABORT),
            TaskSpec(name="other", action="b", agent="Rec"),
        ], {"Rec": agent}).run()

        assert report.aborted
        assert "lit" in report.abort_reason

    @pytest.mark.asyncio
    async def test_degrade_lets_independent_branches_continue(self):
        agent = _Recorder(fail={"a"})
        report = await Pipeline([
            TaskSpec(name="optional", action="a", agent="Rec",
                     on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="independent", action="b", agent="Rec"),
        ], {"Rec": agent}).run()

        assert report.results["optional"].state is TaskState.FAILED
        assert report.results["independent"].state is TaskState.SUCCEEDED
        assert not report.aborted

    @pytest.mark.asyncio
    async def test_retry_recovers_from_a_transient_failure(self):
        agent = _Recorder(fail_times={"a": 2})
        report = await Pipeline([
            TaskSpec(name="flaky", action="a", agent="Rec",
                     on_failure=FailurePolicy.RETRY, max_retries=3,
                     retry_base_delay=0.001),
        ], {"Rec": agent}).run()

        assert report.results["flaky"].state is TaskState.SUCCEEDED
        assert report.results["flaky"].attempts == 3

    @pytest.mark.asyncio
    async def test_clean_is_false_when_anything_was_skipped(self):
        """The property that makes a partial run visibly partial."""
        agent = _Recorder(fail={"a"})
        report = await Pipeline([
            TaskSpec(name="x", action="a", agent="Rec", on_failure=FailurePolicy.DEGRADE),
            TaskSpec(name="y", action="b", agent="Rec", depends_on=("x",)),
        ], {"Rec": agent}).run()
        assert report.clean is False
        assert report.skipped == ["y"]

    @pytest.mark.asyncio
    async def test_unknown_agent_fails_loudly(self):
        report = await Pipeline(
            [TaskSpec(name="x", action="a", agent="Nope")], {},
        ).run()
        assert report.results["x"].state is TaskState.FAILED
        assert "not registered" in report.results["x"].error

    @pytest.mark.asyncio
    async def test_missing_action_fails_loudly(self):
        report = await Pipeline(
            [TaskSpec(name="x", action="nonexistent", agent="Rec")],
            {"Rec": _Recorder()},
        ).run()
        assert "no action" in report.results["x"].error

    @pytest.mark.asyncio
    async def test_sync_actions_are_supported(self):
        report = await Pipeline(
            [TaskSpec(name="x", action="sync_action", agent="Rec")],
            {"Rec": _Recorder()},
        ).run()
        assert report.value("x") == "sync-result"

    @pytest.mark.asyncio
    async def test_condition_skips_without_failing(self):
        agent = _Recorder()
        report = await Pipeline([
            TaskSpec(name="x", action="a", agent="Rec",
                     condition=lambda ctx: False),
        ], {"Rec": agent}).run()
        assert report.results["x"].state is TaskState.SKIPPED
        assert agent.calls == []


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_breaker_refuses_past_the_call_limit(self):
        t = BudgetTracker(limits=BudgetLimits(max_calls=2))
        for _ in range(2):
            t.check()
            t.record("gpt-4o", 100, 50)
        with pytest.raises(BudgetExhausted):
            t.check()

    def test_breaker_refuses_past_the_cost_limit(self):
        t = BudgetTracker(limits=BudgetLimits(max_cost_usd=0.001))
        t.check()
        t.record("claude-opus-5", 10_000, 5_000)
        with pytest.raises(BudgetExhausted):
            t.check()

    def test_oversized_prompt_is_refused(self):
        t = BudgetTracker(limits=BudgetLimits(max_prompt_tokens=100))
        with pytest.raises(BudgetExhausted, match="accumulating"):
            t.check(estimated_prompt_tokens=5_000)

    def test_local_models_cost_nothing_but_are_still_counted(self):
        t = BudgetTracker(limits=BudgetLimits(max_tokens=1_000))
        t.record("gpt-oss-20b", 400, 200)
        assert t.total_cost_usd == 0.0
        assert t.total_tokens == 600

    def test_per_role_attribution(self):
        t = BudgetTracker()
        t.record("gpt-4o", 100, 50, role="reasoning")
        t.record("gpt-4o", 900, 50, role="code")
        assert t.by_role["code"]["tokens"] > t.by_role["reasoning"]["tokens"]
        assert "code" in t.render()

    def test_no_limits_means_never_exhausted(self):
        t = BudgetTracker()
        for _ in range(500):
            t.record("gpt-4o", 1_000, 1_000)
        assert not t.exhausted
        t.check()

    def test_pricing_lookup_is_substring_based(self):
        assert price_for("claude-sonnet-5")[0] > 0
        assert price_for("some-local-llama-70b") == (0.0, 0.0)

    def test_token_estimate_scales_with_content(self):
        short = estimate_tokens([{"content": "hi"}])
        long = estimate_tokens([{"content": "x" * 4000}])
        assert long > short * 100


# ---------------------------------------------------------------------------
# Multiverse
# ---------------------------------------------------------------------------

class TestMultiverse:
    def test_fork_space_is_the_full_grid(self):
        specs = enumerate_specifications()
        assert len(specs) == 4 * 4 * 3 * 2
        assert len({tuple(sorted(s.items())) for s in specs}) == len(specs)

    def test_fragility_is_one_minus_support(self):
        specs = enumerate_specifications()[:10]
        results = [
            SpecificationResult(spec=s, effect=(0.5 if i < 3 else -0.5), p_value=0.01)
            for i, s in enumerate(specs)
        ]
        report = MultiverseReport(results=results)
        assert report.support_rate == pytest.approx(0.3)
        assert report.fragility == pytest.approx(0.7)
        assert not report.robust

    def test_non_significant_effects_are_not_support(self):
        spec = enumerate_specifications()[0]
        r = SpecificationResult(spec=spec, effect=0.5, p_value=0.40)
        assert not r.supports()

    def test_wrong_direction_is_not_support(self):
        spec = enumerate_specifications()[0]
        r = SpecificationResult(spec=spec, effect=-0.5, p_value=0.01)
        assert not r.supports(direction=1)

    def test_robust_requires_no_sign_flips(self):
        specs = enumerate_specifications()[:10]
        results = [SpecificationResult(spec=s, effect=0.5, p_value=0.01) for s in specs]
        results[0].effect = -0.5      # single reversal
        report = MultiverseReport(results=results)
        assert report.sign_flips == 1
        assert not report.robust

    def test_fork_influence_identifies_the_driving_choice(self):
        """The question a specification curve exists to answer."""
        results = []
        for spec in enumerate_specifications():
            effect = 1.0 if spec["outlier_policy"] == "none" else -1.0
            results.append(SpecificationResult(spec=spec, effect=effect, p_value=0.01))
        influence = MultiverseReport(results=results).fork_influence()
        assert max(influence, key=influence.get) == "outlier_policy"

    def test_failed_specifications_do_not_count_as_support(self):
        specs = enumerate_specifications()[:4]
        results = [SpecificationResult(spec=s, error="timed out") for s in specs]
        report = MultiverseReport(results=results)
        assert report.n_ran == 0
        assert report.support_rate == 0.0

    def test_harness_is_injected_above_the_analysis(self):
        code = build_specification_code("print('hi')", {"outlier_policy": "none"})
        assert "multiverse_analyse" in code
        assert code.index("multiverse_analyse") < code.index("print('hi')")

    def test_result_line_is_parsed(self):
        spec = {"outlier_policy": "none"}
        r = parse_specification_result(
            'noise\nSPEC_RESULT:{"effect": 0.42, "p_value": 0.03, "n": 60}', spec,
        )
        assert r.effect == 0.42 and r.n == 60

    def test_missing_result_line_is_an_error_not_a_zero(self):
        r = parse_specification_result("no marker", {"outlier_policy": "none"})
        assert not r.ran and "SPEC_RESULT" in r.error

    def test_nan_effect_is_rejected(self):
        r = parse_specification_result(
            'SPEC_RESULT:{"effect": NaN, "p_value": 0.1}', {"x": "y"},
        )
        assert not r.ran


# ---------------------------------------------------------------------------
# IMRaD
# ---------------------------------------------------------------------------

class TestIMRaD:
    DOC = (
        "Some Title\n\nWe report an inhibitor.\n\n"
        "Introduction\nPrior work suggests X may matter.\n\n"
        "Methods\nCells were cultured at 37C for 48h.\n\n"
        "Results\nViability dropped to 0.43 (p=0.003, n=96).\n\n"
        "Discussion\nThese results might suggest X could be a target.\n\n"
        "References\n1. Smith 2019.\n"
    )

    def test_sections_are_detected_in_order(self):
        found = [s.section for s in imrad.segment(self.DOC)]
        assert found == ["abstract", "introduction", "methods",
                         "results", "discussion", "references"]

    def test_results_outweigh_discussion(self):
        """The distinction the flat chunker erased."""
        assert (imrad.SECTION_WEIGHT[imrad.Section.RESULTS]
                > imrad.SECTION_WEIGHT[imrad.Section.DISCUSSION])

    def test_references_are_excluded_from_indexing(self):
        assert not imrad.should_index(imrad.Section.REFERENCES)
        assert imrad.should_index(imrad.Section.RESULTS)

    def test_hedging_is_penalised(self):
        hedged = "These results might suggest X could potentially be a target."
        firm = "Viability dropped to 0.43 with p equal to 0.003 across 96 wells."
        assert imrad.hedging_density(hedged) > imrad.hedging_density(firm)
        assert (imrad.evidential_score(firm, imrad.Section.RESULTS)
                > imrad.evidential_score(hedged, imrad.Section.RESULTS))

    def test_unheaded_document_degrades_to_unknown(self):
        spans = imrad.segment("Just a blob of extracted PDF text with no headings.")
        assert [s.section for s in spans] == [imrad.Section.UNKNOWN]

    def test_markdown_headings_are_recognised(self):
        assert imrad.detect_heading("## Methods") == imrad.Section.METHODS
        assert imrad.detect_heading("3.1 Results") == imrad.Section.RESULTS

    def test_mixed_section_takes_the_weaker_label(self):
        assert imrad.detect_heading("Results and Discussion") == imrad.Section.DISCUSSION

    def test_grounding_profile_warns_on_speculation_heavy_support(self):
        chunks = [{"section": "discussion"}] * 8 + [{"section": "introduction"}] * 2
        profile = imrad.grounding_profile(chunks)
        assert profile["evidential_fraction"] == 0.0
        assert "speculation" in profile["warning"] or "interpretation" in profile["warning"]


# ---------------------------------------------------------------------------
# Literature hygiene
# ---------------------------------------------------------------------------

class TestLiteratureHygiene:
    def test_doi_extraction_from_various_fields(self):
        assert extract_doi({"doi": "10.1038/s41586-024-1"}) == "10.1038/s41586-024-1"
        assert extract_doi({"url": "https://doi.org/10.1000/XYZ"}) == "10.1000/xyz"
        assert extract_doi({"title": "no identifier here"}) == ""

    def test_preprint_and_published_version_are_one_work(self):
        """Title-equality dedup let both survive and mutually corroborate."""
        assert titles_are_near_duplicates(
            "Novel inhibitor of KRAS in lung cancer",
            "A Novel Inhibitor of KRAS in Lung Cancer",
        )

    def test_unrelated_titles_are_not_merged(self):
        assert not titles_are_near_duplicates(
            "KRAS inhibition in lung cancer", "Gut microbiome and depression",
        )

    def test_dedup_prefers_the_published_version(self):
        report = deduplicate([
            {"title": "Inhibitor of KRAS", "source": "biorxiv", "summary": "a"},
            {"title": "An Inhibitor of KRAS", "doi": "10.1038/x", "source": "pubmed",
             "summary": "a" * 300},
        ])
        assert len(report.kept) == 1
        assert report.kept[0]["source"] == "pubmed"

    def test_same_doi_is_collapsed(self):
        report = deduplicate([
            {"title": "One phrasing", "doi": "10.1038/nature12373"},
            {"title": "Completely different phrasing", "doi": "10.1038/nature12373"},
        ])
        assert len(report.kept) == 1

    def test_canonical_id_prefers_doi_over_url(self):
        assert canonical_id({"doi": "10.1038/nature12373", "url": "http://e.com"}).startswith("doi:")
        assert canonical_id(
            {"url": "https://arxiv.org/abs/2401.12345"}
        ).startswith("arxiv:")

    def test_recent_peer_reviewed_outweighs_old_preprint(self):
        recent = {"published": "2025-01-01", "source": "pubmed"}
        old = {"published": "1998-01-01", "source": "biorxiv"}
        assert quality_weight(recent) > quality_weight(old)

    def test_integrity_flag_downweights_heavily(self):
        base = {"published": "2024-01-01", "source": "pubmed"}
        flagged = dict(base, integrity_flag="expression of concern")
        assert quality_weight(flagged) < quality_weight(base) * 0.6

    def test_apply_hygiene_annotates_and_sorts(self):
        kept, report = apply_hygiene([
            {"title": "Old work", "doi": "10.1038/nature12373", "published": "1999", "source": "arxiv"},
            {"title": "New work", "doi": "10.1126/science.1259855", "published": "2025", "source": "pubmed"},
        ])
        assert kept[0]["title"] == "New work"
        assert all("quality_weight" in p and "canonical_id" in p for p in kept)
        assert report.n_removed == 0

    def test_normalise_title_drops_version_noise(self):
        assert normalise_title("A Novel Method for X") == normalise_title("Method for X")
