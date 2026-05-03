"""
utils/interactive_feedback.py
Helpers for the human-in-the-loop feedback cycle.

The CLI driver (``collect_feedback_cli``) is dependency-injected with a
``prompt`` callable so tests can drive it deterministically without
hooking into ``sys.stdin``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from models.hypothesis import Hypothesis, UserFeedback

logger = logging.getLogger(__name__)


PromptFn = Callable[[str], str]


_VALID_VERDICTS = {"agree", "disagree", "refine", "skip"}


def _normalise_verdict(raw: str) -> str | None:
    """Map common short forms to the canonical verdict.

    Accepted aliases:
        a / agree / yes / y      → agree
        d / disagree / no / n    → disagree
        r / refine / edit / e    → refine
        s / skip / pass          → skip   (treated as "no feedback")
    """
    if raw is None:
        return None
    token = raw.strip().lower()
    if token in {"a", "agree", "yes", "y"}:
        return "agree"
    if token in {"d", "disagree", "no", "n"}:
        return "disagree"
    if token in {"r", "refine", "edit", "e"}:
        return "refine"
    if token in {"s", "skip", "pass", ""}:
        return "skip"
    return None


def collect_feedback_cli(
    hypotheses: Iterable[Hypothesis],
    prompt: PromptFn = input,
    output: Callable[[str], None] = print,
    max_attempts: int = 3,
) -> list[UserFeedback]:
    """Drive an interactive feedback session over a list of hypotheses.

    For each hypothesis the user is asked for a verdict; if the verdict is
    ``refine``, a follow-up free-text comment is collected. Skipped
    hypotheses produce no UserFeedback entry. Invalid verdicts are reprompted
    up to ``max_attempts`` times before defaulting to ``skip``.
    """
    feedbacks: list[UserFeedback] = []
    for i, hyp in enumerate(hypotheses, 1):
        output("")
        output("=" * 70)
        output(f"[{i}] {hyp.title}")
        output(f"    Elo: {hyp.elo_rating:.0f}  |  Novelty: {hyp.novelty_level}")
        if hyp.mechanism:
            output(f"    Mechanism: {hyp.mechanism[:200]}")

        verdict: str | None = None
        for _attempt in range(max_attempts):
            raw = prompt("    Verdict [a]gree / [d]isagree / [r]efine / [s]kip: ")
            verdict = _normalise_verdict(raw)
            if verdict is not None:
                break
            output("    ⚠ Unrecognised input, please try again.")
        if verdict is None:
            verdict = "skip"

        if verdict == "skip":
            continue

        comment = ""
        if verdict == "refine":
            comment = prompt("    Refinement instruction: ").strip()
            if not comment:
                output("    ⚠ Empty refinement — treated as skip.")
                continue

        feedbacks.append(
            UserFeedback(
                hypothesis_id=hyp.id,
                verdict=verdict,
                comment=comment,
            )
        )
    return feedbacks


__all__ = ["collect_feedback_cli", "PromptFn"]
