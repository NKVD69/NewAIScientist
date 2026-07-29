"""
Tests for the Bradley-Terry ranking model and the sandboxed execution wrapper.

Written as regression tests against the specific defects found in the audit:
position bias, missing uncertainty, anti-Darwinian rating inheritance,
undersized tournament budgets, and an AST filter treated as a security
boundary.
"""

from __future__ import annotations

import random

import pytest

from agents.evolution import EvolutionAgent
from agents.ranking import RankingAgent
from models.hypothesis import Hypothesis
from utils import bradley_terry as bt
from utils import sandbox_runner
from utils.safety import check_code_safety

# ---------------------------------------------------------------------------
# Bradley-Terry core
# ---------------------------------------------------------------------------

class TestRatingModel:
    def test_uncertainty_shrinks_with_evidence(self):
        a, b = bt.Rating(), bt.Rating()
        start = a.sigma
        for _ in range(15):
            a, b = bt.update(a, b)
        assert a.sigma < start
        assert a.matches == 15

    def test_uncertainty_never_collapses(self):
        """An LLM judge has irreducible systematic error; σ must not hit zero."""
        a, b = bt.Rating(), bt.Rating()
        for _ in range(500):
            a, b = bt.update(a, b)
        assert a.sigma >= bt.MIN_SIGMA

    def test_winner_gains_loser_loses(self):
        a, b = bt.Rating(), bt.Rating()
        new_a, new_b = bt.update(a, b)
        assert new_a.mu > a.mu
        assert new_b.mu < b.mu

    def test_draw_moves_means_toward_each_other(self):
        a = bt.Rating(mu=1400.0)
        b = bt.Rating(mu=1000.0)
        new_a, new_b = bt.update(a, b, draw=True)
        assert new_a.mu < a.mu and new_b.mu > b.mu
        assert new_a.wins == 0.5 and new_b.wins == 0.5

    def test_low_weight_dampens_the_update(self):
        a, b = bt.Rating(), bt.Rating()
        full_a, _ = bt.update(a, b, weight=1.0)
        weak_a, _ = bt.update(a, b, weight=0.2)
        assert (full_a.mu - a.mu) > (weak_a.mu - a.mu)

    def test_conservative_estimate_penalises_uncertainty(self):
        confident = bt.Rating(mu=1300.0, sigma=50.0)
        lucky = bt.Rating(mu=1350.0, sigma=200.0)
        assert lucky.mu > confident.mu
        # But the confident one ranks higher once uncertainty is priced in.
        assert confident.conservative > lucky.conservative

    def test_win_probability_accounts_for_both_uncertainties(self):
        a = bt.Rating(mu=1400.0, sigma=250.0)
        b = bt.Rating(mu=1200.0, sigma=250.0)
        certain_a = bt.Rating(mu=1400.0, sigma=50.0)
        certain_b = bt.Rating(mu=1200.0, sigma=50.0)
        # Same μ gap; the uncertain pair is closer to a coin flip.
        assert abs(bt.win_probability(a, b) - 0.5) < abs(
            bt.win_probability(certain_a, certain_b) - 0.5
        )


class TestInheritance:
    def test_offspring_is_not_reset_to_the_flat_default(self):
        """The anti-Darwinian bug: children used to start below their parents."""
        parent = bt.Rating(mu=1320.0, sigma=80.0)
        child = bt.inherit(parent)
        assert child.mu > bt.DEFAULT_MU           # not reset to 1200
        assert child.mu < parent.mu               # but regressed toward the mean
        assert child.sigma > parent.sigma         # and less certain

    def test_evolution_agent_applies_inheritance(self):
        parent = Hypothesis(title="parent")
        parent.rating_mu, parent.rating_sigma, parent.rating_matches = 1310.0, 90.0, 12
        child = Hypothesis(title="child")
        EvolutionAgent._inherit_rating(child, parent)

        assert child.rating_mu > bt.DEFAULT_MU
        assert child.rating_matches == 0
        assert child.elo_rating == child.rating_mu

    def test_weak_parent_yields_weak_child(self):
        parent = bt.Rating(mu=900.0, sigma=80.0)
        child = bt.inherit(parent)
        assert child.mu < bt.DEFAULT_MU

    def test_child_of_top_parent_outranks_a_fresh_hypothesis(self):
        """The selection pressure that used to run backwards."""
        parent = bt.Rating(mu=1300.0, sigma=70.0)
        child = bt.inherit(parent)
        fresh = bt.Rating()
        assert child.mu > fresh.mu


class TestPriorsAndBudget:
    def test_prior_reflects_supplied_signals(self):
        strong = bt.prior_from_signals(correctness=0.9, novelty=0.85,
                                       falsifiability=0.9, robustness=0.9)
        weak = bt.prior_from_signals(correctness=0.2, novelty=0.2,
                                     falsifiability=0.1, robustness=0.1)
        assert strong.mu > bt.DEFAULT_MU > weak.mu

    def test_missing_signals_yield_the_neutral_prior(self):
        assert bt.prior_from_signals().mu == pytest.approx(bt.DEFAULT_MU)

    def test_partial_signals_widen_the_prior(self):
        partial = bt.prior_from_signals(correctness=0.8)
        full = bt.prior_from_signals(correctness=0.8, novelty=0.8,
                                     falsifiability=0.8, robustness=0.8)
        assert partial.sigma > full.sigma

    def test_budget_scales_with_pool_size(self):
        """14 hypotheses used to get 12 matches; ~1.7 games each."""
        assert bt.recommended_budget(14) > 50
        assert bt.recommended_budget(4) < bt.recommended_budget(20)

    def test_separation_requires_a_real_gap(self):
        close = {"a": bt.Rating(mu=1210, sigma=180), "b": bt.Rating(mu=1200, sigma=180)}
        clear = {"a": bt.Rating(mu=1600, sigma=50), "b": bt.Rating(mu=1100, sigma=50)}
        assert bt.is_separated(close)[0] is False
        assert bt.is_separated(clear)[0] is True


class TestPairing:
    def test_prefers_uncertain_competitors(self):
        ratings = {
            "settled_a": bt.Rating(mu=1200, sigma=50),
            "settled_b": bt.Rating(mu=1205, sigma=50),
            "unknown": bt.Rating(mu=1200, sigma=200),
        }
        plan = bt.plan_matches(ratings, num_matches=1)
        assert "unknown" in plan.pairs[0]

    def test_avoids_immediate_rematches(self):
        ratings = {c: bt.Rating() for c in "abcd"}
        plan = bt.plan_matches(ratings, num_matches=4)
        assert len(set(frozenset(p) for p in plan.pairs)) >= 3

    def test_returns_nothing_below_two_competitors(self):
        assert bt.plan_matches({"a": bt.Rating()}, num_matches=5).pairs == []


# ---------------------------------------------------------------------------
# Position bias correction
# ---------------------------------------------------------------------------

class _PositionBiasedJudge:
    """A judge that always picks whatever is presented first."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, first: Hypothesis, second: Hypothesis) -> dict[str, str]:
        self.calls += 1
        return {c: first.id for c in
                ("novelty", "plausibility", "testability", "impact")}


class TestPositionBias:
    @pytest.mark.asyncio
    async def test_two_sided_judging_neutralises_a_purely_positional_judge(self):
        agent = RankingAgent(use_local_llm=False)
        agent.llm_client = object()          # pretend an LLM is configured
        judge = _PositionBiasedJudge()
        agent._multi_judge = judge

        a, b = Hypothesis(title="A"), Hypothesis(title="B")
        winner, draw, detail = await agent._judge(a, b)

        assert judge.calls == 2, "pair must be judged in both orders"
        assert draw is True, "a purely positional judge must yield no signal"
        assert winner == ""
        assert "disagreed with itself" in detail

    @pytest.mark.asyncio
    async def test_consistent_judge_still_produces_a_winner(self):
        agent = RankingAgent(use_local_llm=False)
        agent.llm_client = object()
        a, b = Hypothesis(title="A"), Hypothesis(title="B")

        async def always_a(first, second):
            return {c: a.id for c in
                    ("novelty", "plausibility", "testability", "impact")}

        agent._multi_judge = always_a
        winner, draw, _ = await agent._judge(a, b)
        assert winner == a.id and draw is False

    @pytest.mark.asyncio
    async def test_reliability_diagnostic_exposes_the_bias(self):
        agent = RankingAgent(use_local_llm=False)
        agent.llm_client = object()
        agent._multi_judge = _PositionBiasedJudge()
        for _ in range(3):
            await agent._judge(Hypothesis(title="A"), Hypothesis(title="B"))
        assert agent.judge_reliability()["order_invariance_rate"] == 0.0

    def test_exact_tie_is_a_draw_not_a_win_for_a(self):
        """The old code broke ties deterministically toward A."""
        agent = RankingAgent(use_local_llm=False)
        agent.criteria_weights = {"novelty": 0.5, "plausibility": 0.5}
        winner, draw = agent._aggregate_verdicts(
            "a", "b", {"novelty": "a", "plausibility": "b"},
        )
        assert draw is True and winner == ""


class TestRankingAgentIntegration:
    @pytest.mark.asyncio
    async def test_match_updates_both_beliefs_and_mirrors_elo(self):
        agent = RankingAgent(use_local_llm=False, verify_citations=False,
                             rng=random.Random(0))
        a = Hypothesis(title="A", testable_predictions=["p1", "p2", "p3"])
        b = Hypothesis(title="B")

        _, match = await agent.conduct_tournament_match(a, b)

        assert a.rating_matches == 1 and b.rating_matches == 1
        assert a.elo_rating == a.rating_mu       # legacy field stays in sync
        assert match.hypothesis_a_id == a.id

    def test_leaderboard_ranks_conservatively(self):
        agent = RankingAgent(use_local_llm=False, verify_citations=False)
        confident = Hypothesis(title="confident")
        confident.rating_mu, confident.rating_sigma, confident.rating_matches = 1300, 50, 20
        lucky = Hypothesis(title="lucky")
        lucky.rating_mu, lucky.rating_sigma, lucky.rating_matches = 1350, 200, 1

        board = agent.leaderboard([confident, lucky])
        assert board[0]["title"] == "confident"

    def test_prior_is_seeded_from_falsifiability(self):
        agent = RankingAgent(use_local_llm=False, verify_citations=False)
        rigorous = Hypothesis(title="rigorous")
        rigorous.falsifiability_score = 0.95
        vague = Hypothesis(title="vague")
        vague.falsifiability_score = 0.05
        assert agent.get_rating(rigorous).mu > agent.get_rating(vague).mu


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class TestASTFilterIsNotSecurity:
    """Documents that the AST filter cannot be the security boundary."""

    BYPASSES = {
        "importlib": 'import importlib\nimportlib.import_module("os")',
        "dunder_import": '__import__("os").system("id")',
        "http_egress": "import http.client",
        "file_write": 'open("/tmp/x", "w").write("pwned")',
        "exec_compile": 'exec(compile("import os", "<s>", "exec"))',
        "subclasses": "print(().__class__.__mro__[1].__subclasses__())",
        "pickle": "import pickle",
        "infinite_loop": "while True: pass",
        "memory_bomb": "x = bytearray(10**10)",
    }

    @pytest.mark.parametrize("name", sorted(BYPASSES))
    def test_bypass_passes_the_filter(self, name):
        """Each of these defeats the filter — which is why the sandbox exists."""
        is_clean, _ = check_code_safety(self.BYPASSES[name])
        assert is_clean is True

    def test_filter_still_catches_syntax_errors(self):
        assert check_code_safety("def broken(:")[0] is False

    def test_filter_still_flags_direct_os_import(self):
        assert check_code_safety("import os")[0] is False


class TestSandboxRunner:
    def test_detect_runtime_returns_a_known_tier(self):
        assert sandbox_runner.detect_runtime() in ("docker", "podman", "rlimit", "none")

    def test_isolation_report_is_honest_about_strength(self):
        report = sandbox_runner.isolation_report()
        assert set(report) >= {"backend", "strength", "will_execute", "network_enabled"}
        if report["backend"] == "rlimit":
            assert "weak" in report["strength"]

    def test_container_argv_carries_every_hardening_flag(self):
        from pathlib import Path
        argv = sandbox_runner._build_container_argv(
            "docker", sandbox_runner.SandboxPolicy(), Path("/tmp/x"),
        )
        joined = " ".join(argv)
        assert "--network none" in joined
        assert "--read-only" in joined
        assert "--cap-drop ALL" in joined
        assert "--pids-limit" in joined
        assert "--memory" in joined
        assert "--user 65534:65534" in joined

    def test_network_stays_off_unless_explicitly_enabled(self):
        from pathlib import Path
        policy = sandbox_runner.SandboxPolicy(network=False)
        argv = sandbox_runner._build_container_argv("docker", policy, Path("/tmp/x"))
        assert "none" in argv[argv.index("--network") + 1]

    def test_memory_and_swap_capped_together(self):
        """Capping memory without swap lets a bomb page to disk instead."""
        from pathlib import Path
        policy = sandbox_runner.SandboxPolicy(memory_mb=256)
        argv = sandbox_runner._build_container_argv("docker", policy, Path("/tmp/x"))
        assert argv[argv.index("--memory") + 1] == "256m"
        assert argv[argv.index("--memory-swap") + 1] == "256m"

    def test_refuses_to_run_when_no_isolation_is_available(self, monkeypatch):
        monkeypatch.setattr(sandbox_runner, "detect_runtime", lambda *a, **k: "none")
        result = sandbox_runner.run_sandboxed_sync("print('hi')")
        assert result.blocked is True
        assert result.exit_code is None
        assert "Refusing to run" in result.error

    def test_rlimit_requires_explicit_opt_in(self, monkeypatch):
        monkeypatch.setattr(sandbox_runner, "detect_runtime", lambda *a, **k: "rlimit")
        monkeypatch.delenv("NEWAISCI_ALLOW_UNSANDBOXED", raising=False)
        result = sandbox_runner.run_sandboxed_sync("print('hi')")
        assert result.blocked is True

    def test_policy_reads_environment(self, monkeypatch):
        monkeypatch.setenv("NEWAISCI_SANDBOX_MEMORY_MB", "128")
        monkeypatch.setenv("NEWAISCI_SANDBOX_TIMEOUT", "7")
        policy = sandbox_runner.SandboxPolicy.from_env()
        assert policy.memory_mb == 128 and policy.timeout_s == 7

    def test_input_filenames_cannot_escape_the_workdir(self, monkeypatch):
        """A generated path like ../../etc/passwd must be flattened."""
        captured = {}

        def fake_container(runtime, policy, host_dir):
            captured["files"] = sorted(p.name for p in host_dir.iterdir())
            return sandbox_runner.SandboxResult(backend=runtime, exit_code=0)

        monkeypatch.setattr(sandbox_runner, "detect_runtime", lambda *a, **k: "docker")
        monkeypatch.setattr(sandbox_runner, "_run_container", fake_container)

        sandbox_runner.run_sandboxed_sync(
            "print(1)", input_files={"../../etc/passwd": "x"},
        )
        assert captured["files"] == ["experiment.py", "passwd"]
