"""
models/experiment.py — Typed experiments, structured measurements, verdicts.

This module exists to close the epistemic loop that was previously open:
an experiment used to return a free-text blob which the orchestrator then
grep'd for the substring ``"fail"``. Here we give experiments a *kind*
(which constrains what they are allowed to conclude) and a structured
list of ``Measurement`` objects that can be confronted with the
pre-registered ``Prediction`` bundle.

Key invariant enforced downstream (see ``utils.adjudication``):

    A ``DRY_RUN_SIMULATION`` can REFUTE a prediction but can never
    CORROBORATE one.

The rationale: when an LLM generates synthetic data and then runs a
t-test on it, a "significant" result carries zero external information —
it only measures the internal consistency of the generator with itself.
A *contradiction* in such a run is still informative (the hypothesis is
internally incoherent), which is why refutation stays permitted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExperimentKind(Enum):
    """What kind of evidence an experiment run is capable of producing."""

    #: LLM-generated synthetic data. May refute, may NEVER corroborate.
    DRY_RUN_SIMULATION = "simulation"
    #: Analysis of externally-sourced, verifiable data (ChEMBL, DepMap, CSV...).
    REAL_DATA_ANALYSIS = "real_data"
    #: Effect sizes extracted from published literature (meta-analytic).
    LITERATURE_META = "literature_meta"
    #: No data source could be reached; nothing was tested.
    INFEASIBLE = "infeasible"

    @property
    def can_corroborate(self) -> bool:
        """Whether a successful run of this kind may support a hypothesis."""
        return self in (ExperimentKind.REAL_DATA_ANALYSIS,
                        ExperimentKind.LITERATURE_META)

    @property
    def can_refute(self) -> bool:
        """Whether a run of this kind may count against a hypothesis."""
        return self is not ExperimentKind.INFEASIBLE


class VerdictStatus(Enum):
    """Outcome of confronting one pre-registered prediction with one measurement."""

    CORROBORATED = "corroborated"     # measured within threshold, real data
    REFUTED = "refuted"               # measured beyond refuting_threshold
    CONSISTENT_UNSCORED = "consistent_unscored"  # simulation agreed — no credit
    UNTESTED = "untested"             # no measurement matched this prediction
    UNFALSIFIABLE = "unfalsifiable"   # prediction has no usable threshold
    INVALID = "invalid"               # unit mismatch / unusable measurement


@dataclass
class Measurement:
    """A single quantitative observation emitted by an experiment script.

    Produced by parsing the ``RESULTS_JSON:`` line the generated script is
    instructed to print. Field names mirror ``models.hypothesis.Prediction``
    so the two can be matched on ``quantity`` and compared on ``unit``.
    """

    quantity: str = ""
    observed: float = 0.0
    unit: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    test: str = ""
    p_value: float | None = None

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "observed": self.observed,
            "unit": self.unit,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "test": self.test,
            "p_value": self.p_value,
        }


@dataclass
class Verdict:
    """The result of adjudicating one prediction against one measurement."""

    prediction_quantity: str = ""
    status: VerdictStatus = VerdictStatus.UNTESTED
    expected_value: float | None = None
    observed_value: float | None = None
    unit: str = ""
    refuting_threshold: float | None = None
    deviation: float | None = None      # |observed - expected|
    reason: str = ""
    experiment_kind: str = ""

    @property
    def is_refuted(self) -> bool:
        return self.status is VerdictStatus.REFUTED

    @property
    def is_corroborated(self) -> bool:
        return self.status is VerdictStatus.CORROBORATED

    def to_dict(self) -> dict:
        return {
            "quantity": self.prediction_quantity,
            "status": self.status.value,
            "expected": self.expected_value,
            "observed": self.observed_value,
            "unit": self.unit,
            "refuting_threshold": self.refuting_threshold,
            "deviation": self.deviation,
            "reason": self.reason,
            "experiment_kind": self.experiment_kind,
        }


@dataclass
class ExperimentRun:
    """A complete record of one experiment execution.

    Replaces the free-text ``Hypothesis.experimental_results`` string as the
    canonical artefact. The string is still populated for backwards
    compatibility with the UI and the WritingAgent, but all decision logic
    must read this object instead.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    kind: ExperimentKind = ExperimentKind.INFEASIBLE
    measurements: list[Measurement] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    data_source: str = ""               # e.g. "PubChem", "ChEMBL", "synthetic"
    sandbox_backend: str = ""           # "docker" | "podman" | "rlimit" | "none"
    code_sha256: str = ""
    #: The analysis source, retained so ReplicationAgent can re-run it
    #: across the specification grid instead of asking the LLM again.
    code: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ----- Aggregate signals consumed by the ranking / revision cycles -----

    @property
    def n_refuted(self) -> int:
        return sum(1 for v in self.verdicts if v.is_refuted)

    @property
    def n_corroborated(self) -> int:
        return sum(1 for v in self.verdicts if v.is_corroborated)

    @property
    def n_untested(self) -> int:
        return sum(
            1 for v in self.verdicts
            if v.status in (VerdictStatus.UNTESTED, VerdictStatus.INVALID)
        )

    @property
    def refuted(self) -> bool:
        """True iff at least one pre-registered prediction was refuted."""
        return self.n_refuted > 0

    @property
    def evidential_weight(self) -> float:
        """Signed evidence in [-1, 1] contributed by this run.

        Untested predictions contribute nothing (they are *not* treated as
        support, which is the bug the old grep-based logic had). A run that
        tested nothing yields exactly 0.0.
        """
        scored = self.n_refuted + self.n_corroborated
        if scored == 0:
            return 0.0
        return (self.n_corroborated - self.n_refuted) / scored

    def summary(self) -> str:
        """Short human-readable line for logs and the UI."""
        if self.kind is ExperimentKind.INFEASIBLE:
            return f"[{self.kind.value}] not testable — {self.error or 'no data source'}"
        parts = [
            f"[{self.kind.value}]",
            f"{self.n_corroborated} corroborated",
            f"{self.n_refuted} refuted",
            f"{self.n_untested} untested",
        ]
        if not self.kind.can_corroborate and self.verdicts:
            parts.append("(simulation: corroboration not credited)")
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind.value,
            "data_source": self.data_source,
            "sandbox_backend": self.sandbox_backend,
            "code_sha256": self.code_sha256,
            "code": self.code,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "measurements": [m.to_dict() for m in self.measurements],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "n_refuted": self.n_refuted,
            "n_corroborated": self.n_corroborated,
            "n_untested": self.n_untested,
            "evidential_weight": round(self.evidential_weight, 4),
            "error": self.error,
            "timestamp": self.timestamp,
        }


__all__ = [
    "ExperimentKind",
    "ExperimentRun",
    "Measurement",
    "Verdict",
    "VerdictStatus",
]
