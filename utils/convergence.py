"""
utils/convergence.py — ConvergenceTracker for automatic stopping criteria.

Monitors multiple signals across iterations to determine when the
hypothesis exploration has stabilised and further iterations are unlikely
to yield substantially better results.

Tracked signals:
1. Δ Elo — average absolute change in Elo ratings between iterations
2. Diversity — entropy of hypothesis novelty-level distribution
3. Novelty plateau — mean novelty score stagnation
4. Tournament information gain — bits of information gained per match
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceReport:
    """Snapshot of convergence metrics for a single iteration."""
    iteration: int = 0
    delta_elo: float = float("inf")
    diversity_entropy: float = 1.0
    mean_novelty: float = 0.5
    info_gain_per_match: float = 1.0
    converged: bool = False
    reasons: list[str] = field(default_factory=list)


class ConvergenceTracker:
    """Monitors hypothesis exploration convergence across iterations.

    Parameters
    ----------
    elo_threshold
        When the average absolute Elo change between iterations drops
        below this, the Elo signal is "converged". Default 15.
    diversity_min
        Minimum acceptable novelty-distribution entropy. If entropy
        drops below this, diversity has collapsed (bad). Default 0.3.
    novelty_plateau_threshold
        When the absolute change in mean novelty score is below this
        for ``patience`` consecutive iterations, novelty is "plateaued".
        Default 0.02.
    info_gain_threshold
        When the average information gain per tournament match drops
        below this, the tournament signal is "converged". Default 0.1.
    patience
        Number of consecutive iterations ALL signals must be stable
        before ``should_stop()`` returns True. Default 2.
    max_iterations
        Hard upper bound regardless of convergence. Default 10.
    """

    def __init__(
        self,
        elo_threshold: float = 15.0,
        diversity_min: float = 0.3,
        novelty_plateau_threshold: float = 0.02,
        info_gain_threshold: float = 0.1,
        patience: int = 2,
        max_iterations: int = 10,
    ):
        self.elo_threshold = elo_threshold
        self.diversity_min = diversity_min
        self.novelty_plateau_threshold = novelty_plateau_threshold
        self.info_gain_threshold = info_gain_threshold
        self.patience = patience
        self.max_iterations = max_iterations

        # Internal state
        self._history: list[ConvergenceReport] = []
        self._prev_elo_snapshot: dict[str, float] = {}
        self._consecutive_stable: int = 0

    @property
    def history(self) -> list[ConvergenceReport]:
        return list(self._history)

    def update(
        self,
        hypotheses: dict[str, Any],
        tournament_history: list[Any],
        iteration: int,
    ) -> ConvergenceReport:
        """Record a new iteration's metrics and return a ConvergenceReport.

        Parameters
        ----------
        hypotheses
            Dict of ``{id: Hypothesis}``.
        tournament_history
            List of ``TournamentMatch`` objects (cumulative).
        iteration
            Current 1-indexed iteration number.

        Returns
        -------
        A ConvergenceReport with all metrics and a converged flag.
        """
        report = ConvergenceReport(iteration=iteration)

        # 1. Δ Elo
        current_elo = {hid: h.elo_rating for hid, h in hypotheses.items()}
        if self._prev_elo_snapshot:
            common_ids = set(current_elo) & set(self._prev_elo_snapshot)
            if common_ids:
                deltas = [
                    abs(current_elo[hid] - self._prev_elo_snapshot[hid])
                    for hid in common_ids
                ]
                report.delta_elo = sum(deltas) / len(deltas)
            else:
                report.delta_elo = float("inf")
        else:
            report.delta_elo = float("inf")  # First iteration
        self._prev_elo_snapshot = dict(current_elo)

        # 2. Diversity (entropy of novelty_level distribution)
        novelty_levels = [h.novelty_level for h in hypotheses.values()]
        report.diversity_entropy = self._entropy(novelty_levels)

        # 3. Mean novelty
        novelty_scores = []
        for h in hypotheses.values():
            if h.reviews:
                avg_nov = sum(r.novelty_score for r in h.reviews) / len(h.reviews)
                novelty_scores.append(avg_nov)
        report.mean_novelty = (
            sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
        )

        # 4. Information gain per match
        n_matches = len(tournament_history)
        if n_matches > 0 and len(hypotheses) > 1:
            # Simplified: expected information gain = mean |Elo_diff| / 400
            # Normalized to [0, 1] range
            elo_values = list(current_elo.values())
            if len(elo_values) >= 2:
                elo_range = max(elo_values) - min(elo_values)
                report.info_gain_per_match = min(1.0, elo_range / (400 * n_matches))
            else:
                report.info_gain_per_match = 1.0
        else:
            report.info_gain_per_match = 1.0

        # Check convergence signals
        reasons = []
        elo_stable = report.delta_elo < self.elo_threshold
        diversity_ok = report.diversity_entropy >= self.diversity_min
        novelty_stable = self._is_novelty_plateaued(report.mean_novelty)
        info_low = (report.info_gain_per_match < self.info_gain_threshold) or (n_matches == 0)

        if elo_stable:
            reasons.append(f"Elo stable (Δ={report.delta_elo:.1f} < {self.elo_threshold})")
        if novelty_stable:
            reasons.append(f"Novelty plateaued (mean={report.mean_novelty:.3f})")
        if info_low:
            reasons.append(f"Low info gain ({report.info_gain_per_match:.3f})")
        if not diversity_ok:
            reasons.append(
                f"⚠ Diversity collapse (H={report.diversity_entropy:.2f} < {self.diversity_min})"
            )

        # All three positive signals must be True for convergence
        all_stable = elo_stable and novelty_stable and info_low
        if all_stable:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0

        report.converged = self._consecutive_stable >= self.patience
        report.reasons = reasons

        self._history.append(report)

        logger.info(
            "Convergence[%d]: ΔElo=%.1f, H=%.2f, novelty=%.3f, IG=%.3f → %s",
            iteration, report.delta_elo, report.diversity_entropy,
            report.mean_novelty, report.info_gain_per_match,
            "CONVERGED" if report.converged else "CONTINUE",
        )
        return report

    def should_stop(self, iteration: int) -> bool:
        """Return True if the system should stop iterating.

        True when either:
        - The convergence patience is reached, or
        - The hard max_iterations limit is hit.
        """
        if iteration >= self.max_iterations:
            logger.info("Hard iteration limit reached (%d).", self.max_iterations)
            return True
        if self._history and self._history[-1].converged:
            return True
        return False

    def export_history(self) -> list[dict]:
        """Export convergence history as a list of dicts (for JSON serialization)."""
        return [
            {
                "iteration": r.iteration,
                "delta_elo": round(r.delta_elo, 2),
                "diversity_entropy": round(r.diversity_entropy, 3),
                "mean_novelty": round(r.mean_novelty, 3),
                "info_gain_per_match": round(r.info_gain_per_match, 4),
                "converged": r.converged,
                "reasons": r.reasons,
            }
            for r in self._history
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_novelty_plateaued(self, current_novelty: float) -> bool:
        """Check if novelty has been stable across recent iterations."""
        if len(self._history) < 1:
            return False
        prev = self._history[-1].mean_novelty
        return abs(current_novelty - prev) < self.novelty_plateau_threshold

    @staticmethod
    def _entropy(labels: list[str]) -> float:
        """Shannon entropy of a discrete distribution (base 2)."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        ent = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                ent -= p * math.log2(p)
        return ent


__all__ = ["ConvergenceReport", "ConvergenceTracker"]
