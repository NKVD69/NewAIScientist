"""
utils/tournament_pairing.py
Smarter tournament pairing for the Elo-based hypothesis ranker.

Two complementary policies:

1. ``swiss_pairing``: at each round, sort hypotheses by current Elo, walk
   the list in pairs while avoiding rematches we've already seen this
   tournament. Drop-in replacement for ``random.choice`` that
   concentrates comparisons between similarly-rated competitors.

2. ``information_gain_pairing``: prioritise pairs whose Elo expectation
   is closest to 50 % AND whose match history is sparsest. The pair
   maximising ``H(p) - log(1 + n_matches)`` (binary entropy minus a
   sample-count penalty) is returned next, until ``num_matches`` are
   produced.

Both policies are pure / synchronous and operate on lightweight
``(id, elo)`` tuples so they can be unit-tested without spinning up
agents.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

# Public type alias: a hypothesis as seen by the pairer.
Competitor = tuple[str, float]   # (id, elo_rating)
Pair = tuple[str, str]


# ---------------------------------------------------------------------------
# Swiss-system pairing
# ---------------------------------------------------------------------------

def swiss_pairing(
    competitors: Sequence[Competitor],
    history: Iterable[Pair] | None = None,
) -> list[Pair]:
    """One round of Swiss pairing.

    Sort competitors by descending Elo, then pair them top-down while
    avoiding any pair already present in ``history``. When a rematch would
    be unavoidable, swap the lower partner with the next available
    candidate. Odd-numbered fields leave the bottom competitor unpaired.
    """
    if len(competitors) < 2:
        return []

    seen: set[frozenset] = {frozenset(p) for p in (history or [])}
    sorted_by_elo = sorted(competitors, key=lambda c: c[1], reverse=True)
    remaining = [cid for cid, _ in sorted_by_elo]
    pairs: list[Pair] = []

    while len(remaining) >= 2:
        a = remaining.pop(0)
        # Find the next partner that doesn't form a rematch
        partner_idx = None
        for i, cand in enumerate(remaining):
            if frozenset((a, cand)) not in seen:
                partner_idx = i
                break
        # If everyone is a rematch, just take the next-strongest competitor.
        if partner_idx is None:
            partner_idx = 0
        b = remaining.pop(partner_idx)
        pairs.append((a, b))
        seen.add(frozenset((a, b)))

    return pairs


# ---------------------------------------------------------------------------
# Information-gain pairing
# ---------------------------------------------------------------------------

def _expected_score(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def _binary_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p), in bits. H(0)=H(1)=0."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def information_gain_pairing(
    competitors: Sequence[Competitor],
    num_matches: int,
    history: Iterable[Pair] | None = None,
) -> list[Pair]:
    """Greedily select the next ``num_matches`` pairings by information gain.

    For each candidate pair (a, b), the information score is

        IG(a, b) = H(P(a beats b)) - log(1 + n_matches(a, b))

    where ``H`` is binary entropy. The first term peaks when the match is
    a coin-flip (most informative); the second penalises pairs we've
    already tried often this tournament. The greedy version recomputes
    the history bookkeeping after every pick so the same pair isn't
    chosen back-to-back without need.
    """
    if len(competitors) < 2 or num_matches <= 0:
        return []

    counts: dict[frozenset, int] = defaultdict(int)
    for p in (history or []):
        counts[frozenset(p)] += 1

    ids = [cid for cid, _ in competitors]
    elo = dict(competitors)

    selected: list[Pair] = []
    for _ in range(num_matches):
        best_pair: Pair | None = None
        best_score = -math.inf
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                p = _expected_score(elo[a], elo[b])
                ig = _binary_entropy(p) - math.log(1 + counts[frozenset((a, b))])
                if ig > best_score:
                    best_score = ig
                    best_pair = (a, b)
        if best_pair is None:
            break
        selected.append(best_pair)
        counts[frozenset(best_pair)] += 1
    return selected


__all__ = [
    "Competitor",
    "Pair",
    "information_gain_pairing",
    "swiss_pairing",
]
