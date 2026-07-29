"""
agents/ranking.py — RankingAgent: Bayesian Bradley-Terry hypothesis ranking.

Supersedes the previous plain-Elo implementation, which had four defects
that compounded:

1. **Uncorrected position bias.** ``hyp_a`` was always presented as "A", and
   ties broke deterministically toward A. LLM judges have a strong, well
   documented position preference, so the bias was systematic rather than
   averaged out. Fixed here by judging every pair twice with the order
   swapped and keeping only the criteria on which both passes agree.
2. **No uncertainty.** A rating built on one match was indistinguishable
   from one built on twenty, yet it selected which hypothesis got evolved,
   experimented on and written up. Ratings are now Gaussian beliefs (mu, sigma)
   and selection uses the conservative estimate mu - 2*sigma.
3. **No prior.** Review scores, falsifiability and empirical verdicts were
   computed and then discarded. They now seed the prior.
4. **Anti-Darwinian inheritance.** Handled in ``EvolutionAgent`` via
   ``bradley_terry.inherit``.

``Hypothesis.elo_rating`` is kept in sync with ``rating_mu`` so existing
exports, dashboards and sort keys keep working.
"""

from __future__ import annotations

import logging
import random

from models.hypothesis import Hypothesis
from models.memory import TournamentMatch
from utils import bradley_terry as bt
from utils.citation_verifier import (
    verification_score,
    verify_hypothesis,
)
from utils.llm import ensure_str, get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """Tournament ranking via two-sided LLM-as-judge + Bayesian Bradley-Terry."""

    name = "Ranking"

    DEFAULT_CRITERIA_WEIGHTS = {
        "novelty": 0.25,
        "plausibility": 0.30,
        "testability": 0.25,
        "impact": 0.20,
    }

    def __init__(
        self,
        use_local_llm: bool = True,
        verify_citations: bool = True,
        criteria_weights: dict[str, float] | None = None,
        two_sided_judging: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(use_local_llm=use_local_llm)
        self.matches_completed = 0
        self.verify_citations = verify_citations
        self._citation_cache: dict[str, float] = {}
        self.criteria_weights = self._normalise_weights(
            criteria_weights or self.DEFAULT_CRITERIA_WEIGHTS,
        )
        # Judge each pair in both orders and discard criteria where the two
        # passes disagree. Doubles judge cost; this is the single most
        # important signal in the system, so it is worth paying for.
        self.two_sided_judging = two_sided_judging
        self._rng = rng or random.Random()

        # Gaussian beliefs, keyed by hypothesis id.
        self.ratings: dict[str, bt.Rating] = {}
        # Per-criterion beliefs, for the multi-dimensional view.
        self.multi_ratings: dict[str, dict[str, bt.Rating]] = {}
        # Diagnostics on judge self-consistency.
        self.position_bias_stats = {
            "pairs_judged": 0, "criteria_total": 0,
            "criteria_agreed": 0, "draws_from_disagreement": 0,
        }

    # ------------------------------------------------------------------
    # Rating access
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, v) for v in weights.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in weights.items()}

    def get_rating(self, hyp: Hypothesis) -> bt.Rating:
        """Return (and lazily seed) the Gaussian belief for a hypothesis.

        Seeding uses whatever evidence already exists -- review scores,
        falsifiability, empirical support -- rather than a flat 1200. A
        hypothesis carrying its parent's inherited rating keeps it.
        """
        if hyp.id in self.ratings:
            return self.ratings[hyp.id]

        if hyp.rating_matches > 0 or hyp.rating_mu != bt.DEFAULT_MU:
            rating = bt.Rating(
                mu=hyp.rating_mu, sigma=hyp.rating_sigma, matches=hyp.rating_matches,
            )
        else:
            rating = bt.prior_from_signals(
                correctness=self._mean_review(hyp, "correctness_score"),
                novelty=self._mean_review(hyp, "novelty_score"),
                falsifiability=hyp.falsifiability_score or None,
                robustness=(
                    1.0 - hyp.multiverse_fragility
                    if hyp.multiverse_fragility else None
                ),
            )

        self.ratings[hyp.id] = rating
        self._sync_to_hypothesis(hyp, rating)
        return rating

    @staticmethod
    def _mean_review(hyp: Hypothesis, attr: str) -> float | None:
        if not hyp.reviews:
            return None
        vals = [getattr(r, attr, None) for r in hyp.reviews]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    @staticmethod
    def _sync_to_hypothesis(hyp: Hypothesis, rating: bt.Rating) -> None:
        """Mirror the belief onto the hypothesis (elo_rating stays readable)."""
        hyp.rating_mu = rating.mu
        hyp.rating_sigma = rating.sigma
        hyp.rating_matches = rating.matches
        hyp.elo_rating = rating.mu

    def per_criterion_rating(self, hyp_id: str, criterion: str) -> float:
        """Backwards-compatible accessor returning the per-criterion mu."""
        return self.multi_ratings.get(hyp_id, {}).get(criterion, bt.Rating()).mu

    def leaderboard(self, hypotheses: list[Hypothesis]) -> list[dict]:
        """Conservative ranking with credible intervals, for the UI and logs."""
        rows = []
        for hyp in hypotheses:
            r = self.get_rating(hyp)
            low, high = r.credible_interval
            rows.append({
                "id": hyp.id,
                "title": hyp.title,
                "mu": round(r.mu, 1),
                "sigma": round(r.sigma, 1),
                "conservative": round(r.conservative, 1),
                "ci95": (round(low, 1), round(high, 1)),
                "matches": r.matches,
            })
        return sorted(rows, key=lambda d: d["conservative"], reverse=True)

    # ------------------------------------------------------------------
    # Citation trust
    # ------------------------------------------------------------------

    async def _citation_score(self, hyp: Hypothesis) -> float:
        if not self.verify_citations:
            return 1.0
        if hyp.id in self._citation_cache:
            return self._citation_cache[hyp.id]
        try:
            results = await verify_hypothesis(hyp)
            score = verification_score(results)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Citation verification failed for %s: %s", hyp.id, exc)
            score = 1.0
        self._citation_cache[hyp.id] = score
        return score

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------

    async def conduct_tournament_match(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
    ) -> tuple[str, TournamentMatch]:
        """Run one comparison and update both Gaussian beliefs.

        Returns ``(winner_id, match)``. On a draw -- which the two-sided judge
        produces whenever it contradicts itself -- ``winner_id`` is empty.
        """
        rating_a, rating_b = self.get_rating(hyp_a), self.get_rating(hyp_b)

        winner_id, draw, detail = await self._judge(hyp_a, hyp_b)

        cit_a = await self._citation_score(hyp_a)
        cit_b = await self._citation_score(hyp_b)
        # A verdict is only as trustworthy as the weaker set of citations
        # behind it: a win over a hypothesis with fabricated references is
        # not strong evidence either.
        weight = min(cit_a, cit_b)

        if draw or not winner_id:
            new_a, new_b = bt.update(rating_a, rating_b, draw=True, weight=weight)
        elif winner_id == hyp_a.id:
            new_a, new_b = bt.update(rating_a, rating_b, weight=weight)
        else:
            new_b, new_a = bt.update(rating_b, rating_a, weight=weight)

        self.ratings[hyp_a.id], self.ratings[hyp_b.id] = new_a, new_b
        self._sync_to_hypothesis(hyp_a, new_a)
        self._sync_to_hypothesis(hyp_b, new_b)

        summary = self._generate_debate_summary(hyp_a, hyp_b, winner_id, draw, detail)

        match = TournamentMatch(
            hypothesis_a_id=hyp_a.id,
            hypothesis_b_id=hyp_b.id,
            winner_id=winner_id,
            debate_summary=summary,
        )
        self.matches_completed += 1
        return winner_id, match

    # ------------------------------------------------------------------
    # Judging -- position-bias corrected
    # ------------------------------------------------------------------

    async def _judge(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
    ) -> tuple[str, bool, str]:
        """Decide a pair. Returns ``(winner_id, is_draw, detail)``."""
        if self.llm_client:
            try:
                verdicts, detail = await self._consistent_multi_judge(hyp_a, hyp_b)
                if verdicts:
                    self._update_multi_ratings(hyp_a, hyp_b, verdicts)
                    winner, draw = self._aggregate_verdicts(hyp_a.id, hyp_b.id, verdicts)
                    return winner, draw, detail
                # Every criterion was order-dependent => the judge told us
                # nothing about the pair, only about its own position bias.
                self.position_bias_stats["draws_from_disagreement"] += 1
                return "", True, "judge disagreed with itself on every criterion"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Multi-criterion judge failed: %s", exc)

            try:
                winner_id = await self._llm_debate(hyp_a, hyp_b)
                if winner_id:
                    return winner_id, False, "single-criterion fallback judge"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fallback judge failed: %s", exc)

        score_a = self._compute_debate_score(hyp_a)
        score_b = self._compute_debate_score(hyp_b)
        noise_a = self._rng.uniform(0.85, 1.15)
        noise_b = self._rng.uniform(0.85, 1.15)
        winner = hyp_a.id if score_a * noise_a > score_b * noise_b else hyp_b.id
        return winner, False, "heuristic fallback (no LLM)"

    async def _consistent_multi_judge(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
    ) -> tuple[dict[str, str], str]:
        """Judge the pair in both orders; keep only order-invariant verdicts.

        This is the position-bias correction. A criterion where swapping the
        presentation order flips the winner tells us about the judge, not the
        hypotheses, and is dropped rather than counted.
        """
        first = await self._multi_judge(hyp_a, hyp_b)
        if not self.two_sided_judging:
            return first, "single-pass (two-sided judging disabled)"

        # Second pass with the presentation order reversed.
        second_raw = await self._multi_judge(hyp_b, hyp_a)

        self.position_bias_stats["pairs_judged"] += 1
        criteria = set(first) | set(second_raw)
        agreed: dict[str, str] = {}
        flipped: list[str] = []
        for crit in criteria:
            self.position_bias_stats["criteria_total"] += 1
            if crit in first and crit in second_raw and first[crit] == second_raw[crit]:
                agreed[crit] = first[crit]
                self.position_bias_stats["criteria_agreed"] += 1
            else:
                flipped.append(crit)

        detail = (
            f"two-sided: {len(agreed)}/{len(criteria)} criteria order-invariant"
            + (f"; dropped {', '.join(sorted(flipped))}" if flipped else "")
        )
        return agreed, detail

    async def _multi_judge(
        self,
        first: Hypothesis,
        second: Hypothesis,
    ) -> dict[str, str]:
        """One judging pass. ``first`` is presented as A, ``second`` as B.

        Returns ``{criterion: hypothesis_id}`` in terms of the *real* ids, so
        the caller can compare passes regardless of presentation order.
        """
        criteria = list(self.criteria_weights.keys())
        prompt = (
            "You are a senior scientific reviewer comparing two research hypotheses.\n"
            f"Judge each on these four criteria: {', '.join(criteria)}.\n\n"
            "Judge on substance alone. The labels A and B carry no meaning and "
            "the presentation order is arbitrary.\n\n"
            "Hypothesis A:\n"
            f"- Title: {first.title}\n"
            f"- Mechanism: {ensure_str(first.mechanism)[:300]}\n"
            f"- Predictions: {', '.join(first.testable_predictions[:3])}\n\n"
            "Hypothesis B:\n"
            f"- Title: {second.title}\n"
            f"- Mechanism: {ensure_str(second.mechanism)[:300]}\n"
            f"- Predictions: {', '.join(second.testable_predictions[:3])}\n\n"
            "Definitions:\n"
            "- novelty: how much new ground does the hypothesis break?\n"
            "- plausibility: how consistent is it with established science?\n"
            "- testability: how concretely can it be falsified by experiment?\n"
            "- impact: how meaningful would the outcome be if true?\n\n"
            "Return ONLY raw JSON of the form:\n"
            '{"verdicts": {"novelty": "<A or B>", "plausibility": "<A or B>", '
            '"testability": "<A or B>", "impact": "<A or B>"}, '
            '"reasoning": "<one short sentence>"}'
        )
        response = await get_llm_completion(
            self.llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            json_mode=True,
            agent_role="reasoning",
        )
        data = parse_json_response(response.choices[0].message.content)
        raw = data.get("verdicts") or {}

        out: dict[str, str] = {}
        for crit in criteria:
            v = str(raw.get(crit, "")).strip().upper()
            if v == "A" or v == first.id:
                out[crit] = first.id
            elif v == "B" or v == second.id:
                out[crit] = second.id
        return out

    def _aggregate_verdicts(
        self,
        a_id: str,
        b_id: str,
        verdicts: dict[str, str],
    ) -> tuple[str, bool]:
        """Weighted vote over criteria. Returns ``(winner_id, is_draw)``.

        An exact tie is a genuine draw. The old code broke ties toward A,
        which -- since A was always the first element of the pair plan and the
        plan was not shuffled -- injected a systematic bias.
        """
        score_a = sum(
            self.criteria_weights.get(c, 0.0) for c, w in verdicts.items() if w == a_id
        )
        score_b = sum(
            self.criteria_weights.get(c, 0.0) for c, w in verdicts.items() if w == b_id
        )
        if abs(score_a - score_b) < 1e-9:
            return "", True
        return (a_id if score_a > score_b else b_id), False

    def _update_multi_ratings(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        verdicts: dict[str, str],
    ) -> None:
        """Update per-criterion Gaussian beliefs."""
        for criterion, winner in verdicts.items():
            ra = self.multi_ratings.setdefault(hyp_a.id, {}).setdefault(criterion, bt.Rating())
            rb = self.multi_ratings.setdefault(hyp_b.id, {}).setdefault(criterion, bt.Rating())
            if winner == hyp_a.id:
                new_a, new_b = bt.update(ra, rb)
            else:
                new_b, new_a = bt.update(rb, ra)
            self.multi_ratings[hyp_a.id][criterion] = new_a
            self.multi_ratings[hyp_b.id][criterion] = new_b

    async def _llm_debate(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> str | None:
        """Single-criterion fallback judge, with randomised presentation order."""
        swap = self._rng.random() < 0.5
        first, second = (hyp_b, hyp_a) if swap else (hyp_a, hyp_b)

        prompt = f"""
        You are a senior scientific reviewer adjudicating a hypothesis competition.
        Judge on substance; the order of presentation is arbitrary.

        Hypothesis A:
        - Title: {first.title}
        - Mechanism: {ensure_str(first.mechanism)[:300]}
        - Predictions: {', '.join(first.testable_predictions[:3])}
        - Novelty: {first.novelty_level}

        Hypothesis B:
        - Title: {second.title}
        - Mechanism: {ensure_str(second.mechanism)[:300]}
        - Predictions: {', '.join(second.testable_predictions[:3])}
        - Novelty: {second.novelty_level}

        Evaluate on scientific rigor, novelty, feasibility and mechanistic specificity.
        Return JSON: {{"winner": "A" or "B", "reasoning": "one sentence"}}
        """
        response = await get_llm_completion(
            self.llm_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            json_mode=True,
            agent_role="reasoning",
        )
        data = parse_json_response(response.choices[0].message.content)
        raw = str(data.get("winner", data.get("winner_id", ""))).strip().upper()

        if raw == "A" or raw == first.id:
            return first.id
        if raw == "B" or raw == second.id:
            return second.id
        return None

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _compute_debate_score(self, hypothesis: Hypothesis) -> float:
        score = 0.5
        if hypothesis.reviews:
            score = sum(
                r.novelty_score * 0.3 + r.testability_score * 0.3
                + r.correctness_score * 0.2 + r.quality_score * 0.2
                for r in hypothesis.reviews
            ) / len(hypothesis.reviews)
        score += len(hypothesis.testable_predictions) * 0.05
        novelty_bonus = {
            "very_high": 0.15, "high": 0.10, "medium": 0.05, "low": 0.00, "unknown": 0.02,
        }
        score += novelty_bonus.get(hypothesis.novelty_level, 0.02)
        # Empirical adjudication outranks rhetoric: a refuted hypothesis
        # should lose to an untested one.
        score += 0.25 * hypothesis.empirical_support
        return min(1.0, max(0.0, score))

    def _generate_debate_summary(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        winner_id: str,
        draw: bool,
        detail: str,
    ) -> str:
        ra, rb = self.get_rating(hyp_a), self.get_rating(hyp_b)
        if draw or not winner_id:
            return (
                f"Draw between '{hyp_a.title[:40]}' (mu={ra.mu:.0f}+/-{ra.sigma:.0f}) "
                f"and '{hyp_b.title[:40]}' (mu={rb.mu:.0f}+/-{rb.sigma:.0f}). {detail}"
            )
        winner, loser = (hyp_a, hyp_b) if winner_id == hyp_a.id else (hyp_b, hyp_a)
        rw, rl = self.get_rating(winner), self.get_rating(loser)
        return (
            f"'{winner.title[:40]}' (mu={rw.mu:.0f}+/-{rw.sigma:.0f}) defeated "
            f"'{loser.title[:40]}' (mu={rl.mu:.0f}+/-{rl.sigma:.0f}). {detail}"
        )

    # ------------------------------------------------------------------
    # Backwards-compatible shims (DEPRECATED)
    # ------------------------------------------------------------------

    def _update_elo_ratings(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        winner_id: str,
        cit_a: float = 1.0,
        cit_b: float = 1.0,
    ) -> None:
        """DEPRECATED: zero-sum Elo update, kept for legacy callers.

        Delegates to the Bradley-Terry update so external code that still
        calls this keeps working and keeps the beliefs coherent. Note the
        semantics differ deliberately: BT is NOT zero-sum, because the two
        competitors' uncertainties differ and the better-observed one should
        move less. Callers relying on rating-sum conservation should migrate
        to ``conduct_tournament_match``.
        """
        ra, rb = self.get_rating(hyp_a), self.get_rating(hyp_b)
        weight = min(max(0.0, min(1.0, cit_a)), max(0.0, min(1.0, cit_b)))
        if winner_id == hyp_a.id:
            new_a, new_b = bt.update(ra, rb, weight=weight)
        else:
            new_b, new_a = bt.update(rb, ra, weight=weight)
        self.ratings[hyp_a.id], self.ratings[hyp_b.id] = new_a, new_b
        self._sync_to_hypothesis(hyp_a, new_a)
        self._sync_to_hypothesis(hyp_b, new_b)

    def _update_multi_elo(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        verdicts: dict[str, str],
    ) -> None:
        """DEPRECATED alias for :meth:`_update_multi_ratings`."""
        self._update_multi_ratings(hyp_a, hyp_b, verdicts)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def judge_reliability(self) -> dict:
        """How self-consistent the judge is under order permutation.

        An agreement rate near 0.5 means the judge is effectively reading the
        position rather than the content, and the tournament output should not
        be trusted. Surfacing this is the point: the old code could not have
        detected it.
        """
        stats = dict(self.position_bias_stats)
        total = stats["criteria_total"]
        stats["order_invariance_rate"] = (
            stats["criteria_agreed"] / total if total else None
        )
        return stats


__all__ = ["RankingAgent"]
