"""
utils/imrad.py — IMRaD section detection and evidential weighting.

The chunker split papers into ~800-token paragraph blocks with no notion of
section. Downstream, that made a sentence from the **Discussion** — where
authors speculate, in the subjunctive, about what their results *might*
imply — indistinguishable from a sentence in **Results**, where they report
what they measured.

The consequence was not subtle. ``GenerationAgent`` grounds hypotheses in
retrieved chunks and ``ReflectionAgent`` checks plausibility against them, so
author speculation entered the pipeline as established fact and came back out
as "grounding evidence".

This module tags each chunk with its section and an evidential weight, so a
claim can say *where* in the paper its support came from — and so a
hypothesis grounded entirely in other people's speculation can be recognised
as such.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Section:
    """Canonical IMRaD section labels."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    UNKNOWN = "unknown"


#: Evidential weight by section, used when scoring retrieved support.
#:
#: Methods and Results carry what was actually done and measured. Discussion
#: and Introduction carry interpretation and framing — valuable for context,
#: weak as grounding. References are near-useless as retrieval targets and
#: pollute the index with citation strings.
SECTION_WEIGHT: dict[str, float] = {
    Section.RESULTS: 1.00,
    Section.METHODS: 0.95,
    Section.ABSTRACT: 0.80,
    Section.CONCLUSION: 0.60,
    Section.DISCUSSION: 0.55,
    Section.INTRODUCTION: 0.45,
    Section.TITLE: 0.50,
    Section.UNKNOWN: 0.50,
    Section.REFERENCES: 0.05,
}

#: Sections excluded from indexing entirely.
EXCLUDED_SECTIONS = frozenset({Section.REFERENCES})


#: Heading patterns, ordered so that more specific labels win.
_HEADING_PATTERNS: list[tuple[str, re.Pattern]] = [
    (Section.ABSTRACT, re.compile(r"^\s*(abstract|summary)\s*[:.]?\s*$", re.I)),
    (Section.INTRODUCTION, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(introduction|background|related work)\s*[:.]?\s*$", re.I)),
    (Section.METHODS, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(methods?|materials and methods|methodology|"
        r"experimental (?:section|procedures?|setup)|data and methods)\s*[:.]?\s*$", re.I)),
    (Section.RESULTS, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(results?|findings|results and discussion)\s*[:.]?\s*$", re.I)),
    (Section.DISCUSSION, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(discussion|interpretation|limitations)\s*[:.]?\s*$", re.I)),
    (Section.CONCLUSION, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(conclusions?|concluding remarks|"
        r"summary and conclusions?|future work|outlook)\s*[:.]?\s*$", re.I)),
    (Section.REFERENCES, re.compile(
        r"^\s*(?:\d+[.)]?\s*)?(references|bibliography|works cited|"
        r"literature cited)\s*[:.]?\s*$", re.I)),
]

#: "Results and Discussion" is a single section in many journals; its content
#: is mixed, so it gets the lower of the two weights.
_MIXED_HEADING = re.compile(r"results?\s+and\s+discussion", re.I)


def detect_heading(line: str) -> str | None:
    """Return the section a line introduces, or None if it is not a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    if _MIXED_HEADING.search(stripped) and len(stripped) < 40:
        return Section.DISCUSSION      # conservative: treat mixed as the weaker

    for section, pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return section

    # Markdown-style headings ("## Methods", "3.1 Results")
    demoted = re.sub(r"^[#*\s\d.)]+", "", stripped)
    if demoted != stripped and len(demoted) < 40:
        for section, pattern in _HEADING_PATTERNS:
            if pattern.match(demoted):
                return section

    return None


@dataclass
class SectionSpan:
    section: str
    start: int
    end: int

    @property
    def weight(self) -> float:
        return SECTION_WEIGHT.get(self.section, SECTION_WEIGHT[Section.UNKNOWN])


def segment(text: str) -> list[SectionSpan]:
    """Split a full-text document into contiguous IMRaD spans.

    Falls back to a single ``UNKNOWN`` span when no headings are found — many
    PDFs extract as an undifferentiated blob, and pretending otherwise would
    attach confident section labels to guesses.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    boundaries: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        section = detect_heading(line)
        if section:
            boundaries.append((offsets[idx], section))

    if not boundaries:
        return [SectionSpan(Section.UNKNOWN, 0, len(text))]

    spans: list[SectionSpan] = []
    if boundaries[0][0] > 0:
        # Text before the first heading is title + abstract in practice.
        spans.append(SectionSpan(Section.ABSTRACT, 0, boundaries[0][0]))

    for i, (start, section) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        spans.append(SectionSpan(section, start, end))

    return spans


def section_at(spans: list[SectionSpan], offset: int) -> str:
    """Which section a character offset falls in."""
    for span in spans:
        if span.start <= offset < span.end:
            return span.section
    return Section.UNKNOWN


def annotate_chunk(text: str, section: str) -> dict:
    """Metadata to attach to a chunk in the vector store."""
    return {
        "section": section,
        "section_weight": SECTION_WEIGHT.get(section, SECTION_WEIGHT[Section.UNKNOWN]),
        "is_evidential": section in (Section.RESULTS, Section.METHODS),
    }


def should_index(section: str) -> bool:
    """Whether a chunk from this section belongs in the index at all."""
    return section not in EXCLUDED_SECTIONS


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

#: Markers of authorial speculation. Present in Discussion by design; their
#: presence in text presented as evidence is a warning sign.
_HEDGES = re.compile(
    r"\b(may|might|could|possibly|potentially|suggests?|appears? to|"
    r"seems? to|we speculate|we hypothesi[sz]e|it is (?:conceivable|plausible)|"
    r"warrants? further|remains? to be (?:seen|determined|established)|"
    r"future (?:studies|work) (?:will|should))\b",
    re.IGNORECASE,
)


def hedging_density(text: str) -> float:
    """Hedges per 100 words, as a speculation proxy.

    Complements the section label: a "Results" section written entirely in
    the subjunctive is speculation regardless of where it sits.
    """
    words = len((text or "").split())
    if words < 10:
        return 0.0
    return round(100.0 * len(_HEDGES.findall(text)) / words, 2)


def evidential_score(text: str, section: str) -> float:
    """Combined weight in [0, 1]: section provenance × linguistic certainty."""
    base = SECTION_WEIGHT.get(section, SECTION_WEIGHT[Section.UNKNOWN])
    penalty = min(0.4, hedging_density(text) / 10.0)
    return round(max(0.0, base - penalty), 3)


# ---------------------------------------------------------------------------
# Corpus-level diagnostics
# ---------------------------------------------------------------------------

def grounding_profile(chunks: list[dict]) -> dict:
    """Where a hypothesis's supporting chunks actually came from.

    Surfaced to the reviewer. A hypothesis whose support is 90% Introduction
    and Discussion is grounded in other people's speculation — which may be
    fine, but must be visible rather than presented as evidence.
    """
    if not chunks:
        return {"n_chunks": 0, "by_section": {}, "evidential_fraction": 0.0}

    by_section: dict[str, int] = {}
    evidential = 0
    for chunk in chunks:
        section = chunk.get("section", Section.UNKNOWN)
        by_section[section] = by_section.get(section, 0) + 1
        if section in (Section.RESULTS, Section.METHODS):
            evidential += 1

    fraction = evidential / len(chunks)
    return {
        "n_chunks": len(chunks),
        "by_section": dict(sorted(by_section.items(), key=lambda kv: kv[1], reverse=True)),
        "evidential_fraction": round(fraction, 3),
        "warning": (
            "Support is drawn mostly from Introduction/Discussion, i.e. from "
            "authorial interpretation rather than reported measurements."
            if fraction < 0.3 else ""
        ),
    }


__all__ = [
    "EXCLUDED_SECTIONS",
    "SECTION_WEIGHT",
    "Section",
    "SectionSpan",
    "annotate_chunk",
    "detect_heading",
    "evidential_score",
    "grounding_profile",
    "hedging_density",
    "section_at",
    "segment",
    "should_index",
]
