"""
utils/bradley_terry.py — Bayesian Bradley-Terry ranking with explicit uncertainty.

Replaces the plain Elo used by ``RankingAgent``, which had three defects that
compounded each other:

1. **No uncertainty.** A hypothesis with one match and a hypothesis with
   twenty both reported a bare number, so the orchestrator could not tell a
   confident leader from a lucky newcomer. Everything downstream (which
   hypothesis gets evolved, experimented, written up) keyed off that number.
2. **No inheritance.** Evolved hypotheses started at the 1200 default while
   their parents — selected precisely *because* they ranked highest — sat
   above it. Offspring were therefore born below their parents and, with
   ~1.7 matches each, never caught up. Selection pressure ran against
   evolution.
3. **No prior.** Review scores, falsifiability and measured novelty were all
   computed and then discarded; only match outcomes moved the rating.

The model here is a Gaussian-belief Bradley-Terry (the same family as Elo
and TrueSkill, with Elo recoverable as the fixed-variance special case):

    P(a beats b) = σ((μ_a − μ_b) / sqrt(2β² + σ_a² + σ_b²))

Beliefs update by moment matching after each result, so σ shrinks as
evidence accumulates. Draws are first-class, which matters because the
position-bias correction in ``RankingAgent`` produces genuine ties whenever
the two orderings of a pair disagree.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Rating scale, kept Elo-like so existing dashboards remain readable.
DEFAULT_MU = 1200.0
DEFAULT_SIGMA = 200.0

#: Per-comparison noise. Large, because an LLM judge is a noisy instrument:
#: even after two-sided position correction, repeat verdicts on the same pair
#: disagree often. Understating β manufactures false confidence.
DEFAULT_BETA = 220.0

#: Floor on σ. Prevents the model from claiming near-certainty from a judge
#: whose systematic error is irreducible.
MIN_SIGMA = 45.0

#: Per-update inflation, so beliefs stay responsive when a hypothesis is
#: revised mid-tournament (its true strength genuinely changed).
DEFAULT_TAU = 6.0


@dataclass
class Rating:
    """Gaussian belief about a hypothesis's latent strength."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA
    matches: int = 0
    wins: float = 0.0          # fractional: a draw contributes 0.5
    losses: float = 0.0

    @property
    def conservative(self) -> float:
        """Rank-safe point estimate: μ − 2σ.

        Ranking on this rather than on μ is what stops a hypothesis with one
        lucky win from being handed the manuscript.
        """
        return self.mu - 2.0 * self.sigma

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Approximate 95% credible interval on the latent strength."""
        return (self.mu - 1.96 * self.sigma, self.mu + 1.96 * self.sigma)

    def to_dict(self) -> dict:
        low, high = self.credible_interval
        return {
            "mu": round(self.mu, 1),
            "sigma": round(self.sigma, 1),
            "conservative": round(self.conservative, 1),
            "ci95_low": round(low, 1),
            "ci95_high": round(high, 1),
            "matches": self.matches,
            "wins": self.wins,
            "losses": self.losses,
        }


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def win_probability(a: Rating, b: Rating, beta: float = DEFAULT_BETA) -> float:
    """P(a beats b), marginalising over both beliefs.

    The uncertainty of *both* competitors enters the denominator, so a
    match between two poorly-observed hypotheses is correctly predicted as
    close to a coin flip regardless of their μ gap.
    """
    denom = math.sqrt(2.0 * beta * beta + a.sigma ** 2 + b.sigma ** 2)
    return _logistic((a.mu - b.mu) / max(denom, 1e-9))


def expected_information(a: Rating, b: Rating, beta: float = DEFAULT_BETA) -> float:
    """Binary entropy of the predicted outcome, in bits.

    A pairing worth playing is one whose result we cannot already guess:
    entropy peaks at p = 0.5 and vanishes at a foregone conclusion.
    """
    p = win_probability(a, b, beta)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def update(
    winner: Rating,
    loser: Rating,
    draw: bool = False,
    beta: float = DEFAULT_BETA,
    tau: float = DEFAULT_TAU,
    weight: float = 1.0,
) -> tuple[Rating, Rating]:
    """Moment-matched Bayesian update after one comparison.

    Parameters
    ----------
    winner, loser
        Ratings to update. On a draw the labels are arbitrary.
    draw
        True when the two-sided judge disagreed with itself and the pair was
        scored as a tie.
    weight
        Confidence in this observation, in [0, 1]. ``RankingAgent`` passes the
        citation-verification score here, so a win built on hallucinated
        references moves the ratings less.

    Returns the two updated ``Rating`` objects (new instances; inputs are not
    mutated).
    """
    weight = max(0.0, min(1.0, weight))

    # Dynamics: inflate variance slightly so old beliefs stay revisable.
    s_w = math.sqrt(winner.sigma ** 2 + tau ** 2)
    s_l = math.sqrt(loser.sigma ** 2 + tau ** 2)

    c2 = 2.0 * beta * beta + s_w ** 2 + s_l ** 2
    c = math.sqrt(max(c2, 1e-9))

    expected_w = _logistic((winner.mu - loser.mu) / c)
    actual_w = 0.5 if draw else 1.0
    surprise = (actual_w - expected_w) * weight

    # Mean update, scaled by each competitor's share of the total variance:
    # the less certain we were about a hypothesis, the more its mean moves.
    new_mu_w = winner.mu + (s_w ** 2 / c) * surprise
    new_mu_l = loser.mu - (s_l ** 2 / c) * surprise

    # Variance update. The information gained is largest for informative
    # (near-even) matches and near-zero for foregone conclusions.
    info = expected_w * (1.0 - expected_w)
    shrink_w = 1.0 - (s_w ** 2 / c2) * info * weight
    shrink_l = 1.0 - (s_l ** 2 / c2) * info * weight

    new_sigma_w = max(MIN_SIGMA, s_w * math.sqrt(max(shrink_w, 1e-4)))
    new_sigma_l = max(MIN_SIGMA, s_l * math.sqrt(max(shrink_l, 1e-4)))

    w_win = 0.5 if draw else 1.0
    return (
        Rating(
            mu=new_mu_w, sigma=new_sigma_w,
            matches=winner.matches + 1,
            wins=winner.wins + w_win,
            losses=winner.losses + (1.0 - w_win),
        ),
        Rating(
            mu=new_mu_l, sigma=new_sigma_l,
            matches=loser.matches + 1,
            wins=loser.wins + (1.0 - w_win),
            losses=loser.losses + w_win,
        ),
    )


# ---------------------------------------------------------------------------
# Priors, inheritance, budgets, stopping
# ---------------------------------------------------------------------------

def prior_from_signals(
    correctness: float | None = None,
    novelty: float | None = None,
    falsifiability: float | None = None,
    robustness: float | None = None,
    spread: float = 400.0,
) -> Rating:
    """Build an informative prior from signals the system already computes.

    ``correctness`` and ``novelty`` come from the review committee,
    ``falsifiability`` from the PreregistrationAgent, ``robustness`` from the
    multiverse replication. All were previously computed and then ignored —
    only match outcomes moved a rating. Folding them into the prior means a
    hypothesis with three quantified, falsifiable predictions does not start
    level with one whose predictions are unfalsifiable prose.

    Missing signals are skipped rather than defaulted, so a hypothesis with
    no reviews yet gets the neutral prior instead of a fabricated one.
    """
    weights = {
        "correctness": (correctness, 0.35),
        "novelty": (novelty, 0.25),
        "falsifiability": (falsifiability, 0.25),
        "robustness": (robustness, 0.15),
    }
    present = {k: (v, w) for k, (v, w) in weights.items() if v is not None}
    if not present:
        return Rating()

    total_w = sum(w for _, w in present.values())
    score = sum(max(0.0, min(1.0, v)) * w for v, w in present.values()) / total_w

    # Confidence in the prior scales with how much of the evidence we have.
    coverage = total_w / sum(w for _, w in weights.values())
    sigma = DEFAULT_SIGMA * (1.0 - 0.25 * coverage)

    return Rating(mu=DEFAULT_MU + spread * (score - 0.5), sigma=sigma)


def inherit(parent: Rating, regression: float = 0.7, sigma_inflation: float = 1.6) -> Rating:
    """Derive an offspring rating from its parent.

    Fixes the anti-Darwinian defect: offspring used to start at the 1200
    default while their parents — selected for being top-ranked — sat above
    it, so evolution was penalised by construction.

    Inheritance is partial (``regression`` < 1) because an evolved hypothesis
    is a genuinely different object, and its σ is inflated because we know
    less about it than about its parent. A child therefore starts near its
    parent's μ but with wide error bars, which the information-gain pairer
    then prioritises for testing.
    """
    mu = DEFAULT_MU + regression * (parent.mu - DEFAULT_MU)
    sigma = min(DEFAULT_SIGMA, parent.sigma * sigma_inflation)
    return Rating(mu=mu, sigma=max(sigma, MIN_SIGMA))


def recommended_budget(n_competitors: int, floor: int = 8) -> int:
    """Matches needed for the ranking to be identified.

    Pairwise ranking needs Θ(n log n) comparisons. The shipped defaults gave
    12 matches for 14 hypotheses — 1.7 games each, where ~53 are needed —
    so the reported ordering was mostly noise, yet it selected the hypothesis
    that got written up.
    """
    n = max(2, int(n_competitors))
    return max(floor, int(round(2.0 * n * math.log2(n))))


def is_separated(
    ratings: dict[str, Rating],
    top_k: int = 1,
    n_sigma: float = 2.0,
) -> tuple[bool, str]:
    """Whether the top-``k`` cut is statistically distinguishable.

    Used as a stopping rule: keep playing until the leader is separated from
    the runner-up by ``n_sigma`` of their combined uncertainty. This lets the
    tournament stop early when the answer is clear, and — more importantly —
    keep going when it is not, instead of stopping at a fixed match count.
    """
    if len(ratings) < 2:
        return True, "fewer than two competitors"

    ordered = sorted(ratings.values(), key=lambda r: r.mu, reverse=True)
    k = max(1, min(top_k, len(ordered) - 1))
    leader, challenger = ordered[k - 1], ordered[k]

    gap = leader.mu - challenger.mu
    combined = math.sqrt(leader.sigma ** 2 + challenger.sigma ** 2)
    separated = gap > n_sigma * combined

    detail = (
        f"gap={gap:.1f}, {n_sigma}σ threshold={n_sigma * combined:.1f} "
        f"(σ_leader={leader.sigma:.1f}, σ_challenger={challenger.sigma:.1f})"
    )
    return separated, detail


def rank(ratings: dict[str, Rating], conservative: bool = True) -> list[tuple[str, Rating]]:
    """Order competitors, conservatively (μ − 2σ) by default."""
    key = (lambda item: item[1].conservative) if conservative else (lambda item: item[1].mu)
    return sorted(ratings.items(), key=key, reverse=True)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dataclass
class PairingPlan:
    pairs: list[tuple[str, str]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def plan_matches(
    ratings: dict[str, Rating],
    num_matches: int,
    history: list[tuple[str, str]] | None = None,
    beta: float = DEFAULT_BETA,
) -> PairingPlan:
    """Greedily select the most informative unplayed pairs.

    Utility = expected information (bits) × uncertainty of the pair, minus a
    rematch penalty. Weighting by σ pushes the tournament toward hypotheses
    we know least about — which is exactly where newly-evolved offspring sit,
    so they now get tested instead of languishing unplayed.
    """
    plan = PairingPlan()
    ids = list(ratings.keys())
    if len(ids) < 2:
        return plan

    played: dict[frozenset, int] = {}
    for a, b in (history or []):
        played[frozenset((a, b))] = played.get(frozenset((a, b)), 0) + 1

    working = {k: Rating(**vars(v)) for k, v in ratings.items()}

    for _ in range(max(0, num_matches)):
        best: tuple[float, str, str] | None = None
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = working[a_id], working[b_id]
                bits = expected_information(a, b, beta)
                uncertainty = (a.sigma + b.sigma) / (2.0 * DEFAULT_SIGMA)
                rematch = played.get(frozenset((a_id, b_id)), 0)
                utility = bits * (0.5 + uncertainty) - 0.75 * rematch
                if best is None or utility > best[0]:
                    best = (utility, a_id, b_id)

        if best is None:
            break

        _, a_id, b_id = best
        plan.pairs.append((a_id, b_id))
        plan.rationale.append(
            f"{a_id} vs {b_id}: {expected_information(working[a_id], working[b_id], beta):.2f} bits, "
            f"σ=({working[a_id].sigma:.0f}, {working[b_id].sigma:.0f})"
        )
        played[frozenset((a_id, b_id))] = played.get(frozenset((a_id, b_id)), 0) + 1

        # Anticipate the information this match will yield so the next
        # selection does not pile onto the same pair.
        working[a_id], working[b_id] = update(
            working[a_id], working[b_id], draw=True, beta=beta,
        )

    return plan


__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_MU",
    "DEFAULT_SIGMA",
    "PairingPlan",
    "Rating",
    "expected_information",
    "inherit",
    "is_separated",
    "plan_matches",
    "prior_from_signals",
    "rank",
    "recommended_budget",
    "update",
    "win_probability",
]
