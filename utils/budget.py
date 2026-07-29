"""
utils/budget.py — Token and cost budgeting with a circuit breaker.

The system had none. ``utils/llm.py`` *counted* tokens into
``_llm_state["total_tokens"]`` and never acted on the number.

Volume for one default run: ``ReflectionAgent._review_with_llm`` issues five
calls per hypothesis per review (entity extraction + three reviewers + meta
reviewer). At 14 hypotheses that is 70 calls for reviews alone, before
generation, tournament (now two-sided, so doubled), evolution,
pre-registration, experiments, replication and a six-section manuscript. A
full cycle comfortably clears 200 calls with no ceiling, no cost cap and no
breaker — a misconfigured loop or a runaway retry storm bills silently until
someone notices.

The tracker is deliberately advisory-then-hard: it warns at thresholds,
then refuses. Refusal raises ``BudgetExhausted`` so the caller sees an
explicit failure rather than a quietly truncated run.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    """Raised when a call would exceed the configured budget."""


#: Approximate USD per 1M tokens, by model-name substring. Local models are
#: free but still consume wall-clock and context, so they are tracked too.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # substring: (input_per_1M, output_per_1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4": (30.00, 60.00),
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    "gpt-oss": (0.0, 0.0),      # local via LM Studio
    "mistral": (0.0, 0.0),
    "llama": (0.0, 0.0),
    "qwen": (0.0, 0.0),
}


def price_for(model_name: str) -> tuple[float, float]:
    """Return ``(input, output)`` USD per 1M tokens for a model name."""
    lowered = (model_name or "").lower()
    for key, prices in DEFAULT_PRICING.items():
        if key in lowered:
            return prices
    return (0.0, 0.0)


@dataclass
class BudgetLimits:
    """Ceilings for one session. ``None`` means unlimited for that dimension."""

    max_tokens: int | None = None
    max_calls: int | None = None
    max_cost_usd: float | None = None
    #: Fraction of a limit at which a warning is emitted (once per dimension).
    warn_at: float = 0.8
    #: Refuse any single call whose prompt alone exceeds this. Catches runaway
    #: context accumulation before it becomes expensive.
    max_prompt_tokens: int | None = None

    @classmethod
    def from_env(cls) -> BudgetLimits:
        import os

        def _opt_int(name: str) -> int | None:
            raw = os.environ.get(name, "").strip()
            try:
                return int(raw) if raw else None
            except ValueError:
                return None

        def _opt_float(name: str) -> float | None:
            raw = os.environ.get(name, "").strip()
            try:
                return float(raw) if raw else None
            except ValueError:
                return None

        return cls(
            max_tokens=_opt_int("NEWAISCI_MAX_TOKENS"),
            max_calls=_opt_int("NEWAISCI_MAX_LLM_CALLS"),
            max_cost_usd=_opt_float("NEWAISCI_MAX_COST_USD"),
            max_prompt_tokens=_opt_int("NEWAISCI_MAX_PROMPT_TOKENS"),
        )


@dataclass
class BudgetTracker:
    """Thread-safe accounting with a hard circuit breaker.

    Per-role and per-agent breakdowns exist so a run that overspends can be
    attributed. In practice the tournament and the review committee dominate,
    and knowing which is which is what makes the trade-off decidable.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0
    refused_calls: int = 0
    by_role: dict[str, dict] = field(default_factory=dict)
    _warned: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Breaker
    # ------------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        """Whether any hard limit has been reached."""
        lim = self.limits
        if lim.max_tokens is not None and self.total_tokens >= lim.max_tokens:
            return True
        if lim.max_calls is not None and self.total_calls >= lim.max_calls:
            return True
        if lim.max_cost_usd is not None and self.total_cost_usd >= lim.max_cost_usd:
            return True
        return False

    def check(self, estimated_prompt_tokens: int = 0, role: str = "default") -> None:
        """Raise ``BudgetExhausted`` if this call must not proceed.

        Called before every LLM request. Deliberately raises rather than
        returning a flag: a silently skipped call produces a half-built
        artefact that looks complete.
        """
        lim = self.limits

        if (lim.max_prompt_tokens is not None
                and estimated_prompt_tokens > lim.max_prompt_tokens):
            with self._lock:
                self.refused_calls += 1
            raise BudgetExhausted(
                f"prompt of ~{estimated_prompt_tokens} tokens exceeds the "
                f"per-call cap of {lim.max_prompt_tokens} (role={role}). "
                "Context is probably accumulating unboundedly."
            )

        if self.exhausted:
            with self._lock:
                self.refused_calls += 1
            raise BudgetExhausted(
                f"budget exhausted before call (role={role}): {self.summary()}"
            )

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        role: str = "default",
    ) -> None:
        """Record one completed call."""
        in_price, out_price = price_for(model)
        cost = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0

        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            self.total_calls += 1
            self.total_cost_usd += cost

            bucket = self.by_role.setdefault(
                role, {"calls": 0, "tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["tokens"] += prompt_tokens + completion_tokens
            bucket["cost_usd"] += cost

        self._maybe_warn()

    def _maybe_warn(self) -> None:
        lim = self.limits
        checks = [
            ("tokens", self.total_tokens, lim.max_tokens),
            ("calls", self.total_calls, lim.max_calls),
            ("cost", self.total_cost_usd, lim.max_cost_usd),
        ]
        for label, used, cap in checks:
            if cap is None or label in self._warned:
                continue
            if used >= cap * lim.warn_at:
                self._warned.add(label)
                logger.warning(
                    "Budget warning: %s at %.0f%% of limit (%s / %s).",
                    label, 100.0 * used / cap, used, cap,
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lim = self.limits
        parts = [
            f"{self.total_calls} calls"
            + (f"/{lim.max_calls}" if lim.max_calls else ""),
            f"{self.total_tokens} tokens"
            + (f"/{lim.max_tokens}" if lim.max_tokens else ""),
        ]
        if self.total_cost_usd > 0 or lim.max_cost_usd:
            parts.append(
                f"${self.total_cost_usd:.4f}"
                + (f"/${lim.max_cost_usd:.2f}" if lim.max_cost_usd else "")
            )
        if self.refused_calls:
            parts.append(f"{self.refused_calls} refused")
        return " · ".join(parts)

    def render(self) -> str:
        """Per-role attribution — which agents actually spent the budget."""
        lines = ["LLM budget", "─" * 60, self.summary(), ""]
        if not self.by_role:
            lines.append("  (no calls recorded)")
            return "\n".join(lines)

        ordered = sorted(
            self.by_role.items(), key=lambda kv: kv[1]["tokens"], reverse=True,
        )
        for role, stats in ordered:
            share = (
                100.0 * stats["tokens"] / self.total_tokens
                if self.total_tokens else 0.0
            )
            lines.append(
                f"  {role:<14} {stats['calls']:>4} calls  "
                f"{stats['tokens']:>8} tokens ({share:4.1f}%)  "
                f"${stats['cost_usd']:.4f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_calls = 0
            self.total_cost_usd = 0.0
            self.refused_calls = 0
            self.by_role.clear()
            self._warned.clear()


# ---------------------------------------------------------------------------
# Session-global tracker
# ---------------------------------------------------------------------------

_ACTIVE: BudgetTracker | None = None


def get_budget() -> BudgetTracker | None:
    """Return the active tracker, if budgeting is enabled."""
    return _ACTIVE


def set_budget(tracker: BudgetTracker | None) -> None:
    """Install (or clear) the session-wide tracker."""
    global _ACTIVE
    _ACTIVE = tracker


def enable_from_env() -> BudgetTracker | None:
    """Install a tracker from environment variables, if any limit is set."""
    limits = BudgetLimits.from_env()
    if all(v is None for v in (limits.max_tokens, limits.max_calls,
                               limits.max_cost_usd, limits.max_prompt_tokens)):
        return None
    tracker = BudgetTracker(limits=limits)
    set_budget(tracker)
    logger.info("LLM budgeting enabled: %s", tracker.summary())
    return tracker


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap prompt-size estimate (~4 chars/token), no tokenizer needed.

    Used only for the pre-call guard, where an order of magnitude is enough.
    Exact accounting comes from the API's usage field afterwards.
    """
    chars = sum(len(str(m.get("content", ""))) for m in messages or [])
    return max(1, chars // 4)


__all__ = [
    "BudgetExhausted",
    "BudgetLimits",
    "BudgetTracker",
    "enable_from_env",
    "estimate_tokens",
    "get_budget",
    "price_for",
    "set_budget",
]
