"""
utils/literature_refresh.py
Helpers for incremental literature refresh: filter a freshly-fetched paper
list down to those that (a) are newer than the per-source watermark, AND
(b) are not yet present in the local set, and update the watermark.

Pure functions — no agent / network dependencies — so they're trivially
unit-tested.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

logger = logging.getLogger(__name__)


def _parse_iso(ts: str | None) -> datetime | None:
    """Tolerant ISO-8601 parser. Accepts trailing 'Z' and date-only strings."""
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Try common date-only forms
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


def _paper_key(paper: dict) -> str:
    """Stable identity for a paper. Prefers URL, then title."""
    return paper.get("url") or paper.get("doi") or paper.get("title") or ""


def filter_new_papers(
    fetched: Iterable[dict],
    existing: Iterable[dict],
    last_seen: str | None = None,
) -> list[dict]:
    """Return papers in ``fetched`` that are both unseen and newer than the watermark.

    A paper is considered "new" iff:
      - its identity key is not present in ``existing``, AND
      - its ``published`` timestamp (when parseable) is strictly after
        ``last_seen`` (when ``last_seen`` is provided).

    Papers without a parseable timestamp are kept (we err on the side of
    keeping potentially-new content rather than dropping it silently).
    """
    seen_keys = {_paper_key(p) for p in existing if _paper_key(p)}
    cutoff = _parse_iso(last_seen)
    out: list[dict] = []
    for paper in fetched or []:
        key = _paper_key(paper)
        if key and key in seen_keys:
            continue
        if cutoff is not None:
            published = _parse_iso(paper.get("published") or paper.get("date"))
            if published is not None and published <= cutoff:
                continue
        out.append(paper)
    return out


def update_watermark(
    last_seen: dict[str, str],
    source: str,
    papers: Iterable[dict],
) -> dict[str, str]:
    """Advance ``last_seen[source]`` to the latest ``published`` timestamp seen.

    Mutates and returns the same dict for chaining. If no parseable
    timestamp is found, ``last_seen`` is left untouched.
    """
    latest: datetime | None = _parse_iso(last_seen.get(source))
    for paper in papers or []:
        ts = _parse_iso(paper.get("published") or paper.get("date"))
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    if latest is not None:
        last_seen[source] = latest.isoformat()
    return last_seen


__all__ = ["filter_new_papers", "update_watermark"]
