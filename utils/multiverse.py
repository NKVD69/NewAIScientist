"""
utils/multiverse.py — Specification-curve analysis (multiverse replication).

Replaces seed-variation "replication", which measured nothing:

* On real, deterministic data (a CSV, a ChEMBL pull), re-running with a
  different ``np.random.seed`` gives **identical** output. The system
  reported perfect reproducibility — true, and completely uninformative.
* On synthetic data, it measured the variance of the pseudo-random generator
  the LLM had just written. A "robust" result meant the model had chosen a
  small standard deviation.

The real reproducibility question is not *does this rerun the same way?* but
*does the conclusion survive the analytic choices the analyst could
defensibly have made otherwise?* That is a specification curve (Simonsohn,
Simmons & Nelson) / multiverse analysis (Steegen et al.): enumerate the
defensible forks, run all of them, and report the distribution of effects
rather than a single point.

The fragility this produces feeds ``Hypothesis.multiverse_fragility``, which
the Bradley-Terry prior consumes — so a finding that only survives 4 of 54
defensible specifications is penalised in the ranking rather than presented
as a result.
"""

from __future__ import annotations

import itertools
import logging
import math
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The fork space
# ---------------------------------------------------------------------------

#: Analytic choices that are individually defensible and jointly arbitrary.
#: Each is a genuine fork an analyst faces and rarely pre-registers.
DEFAULT_FORKS: dict[str, list[str]] = {
    "outlier_policy": ["none", "iqr_1.5", "winsorize_5pct", "drop_3sd"],
    "test": ["welch_t", "student_t", "mann_whitney", "permutation"],
    "transform": ["identity", "log", "rank"],
    "covariate_adjust": ["none", "adjusted"],
}


def enumerate_specifications(
    forks: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    """Cartesian product of the fork space.

    The defaults give 4 × 4 × 3 × 2 = 96 specifications. That is the point:
    the number of defensible analyses is large, and reporting one of them as
    *the* result is a choice that usually goes unexamined.
    """
    forks = forks or DEFAULT_FORKS
    keys = sorted(forks)
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(forks[k] for k in keys))
    ]


def specification_id(spec: dict[str, str]) -> str:
    return "|".join(f"{k}={spec[k]}" for k in sorted(spec))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SpecificationResult:
    """Outcome of one analytic specification."""

    spec: dict[str, str] = field(default_factory=dict)
    effect: float | None = None
    p_value: float | None = None
    n: int | None = None
    error: str = ""

    @property
    def ran(self) -> bool:
        return self.effect is not None and not self.error

    def supports(self, direction: int = 1, alpha: float = 0.05) -> bool:
        """Whether this specification supports the hypothesised direction.

        Requires both a same-signed effect and significance: an effect in the
        right direction that fails to reach significance is not support, and
        a significant effect in the wrong direction is evidence against.
        """
        if not self.ran:
            return False
        if self.p_value is not None and self.p_value > alpha:
            return False
        return (self.effect or 0.0) * direction > 0


@dataclass
class MultiverseReport:
    """Distribution of effects across the whole specification space."""

    results: list[SpecificationResult] = field(default_factory=list)
    direction: int = 1
    alpha: float = 0.05

    # -- Population --------------------------------------------------------

    @property
    def ran(self) -> list[SpecificationResult]:
        return [r for r in self.results if r.ran]

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_ran(self) -> int:
        return len(self.ran)

    @property
    def n_supporting(self) -> int:
        return sum(1 for r in self.ran if r.supports(self.direction, self.alpha))

    # -- Headline metrics --------------------------------------------------

    @property
    def support_rate(self) -> float:
        """Fraction of executed specifications that support the hypothesis."""
        return self.n_supporting / self.n_ran if self.n_ran else 0.0

    @property
    def fragility(self) -> float:
        """Fraction of specifications that do NOT support — in [0, 1].

        Fed to ``Hypothesis.multiverse_fragility``. A result surviving 4 of 54
        forks has fragility 0.93 and should not be written up as a finding.
        """
        return 1.0 - self.support_rate

    @property
    def effects(self) -> list[float]:
        return [r.effect for r in self.ran if r.effect is not None]

    @property
    def median_effect(self) -> float | None:
        return statistics.median(self.effects) if self.effects else None

    @property
    def effect_range(self) -> tuple[float, float] | None:
        return (min(self.effects), max(self.effects)) if self.effects else None

    @property
    def sign_flips(self) -> int:
        """How many specifications reverse the sign of the effect."""
        if not self.effects:
            return 0
        med = self.median_effect or 0.0
        ref = 1 if med >= 0 else -1
        return sum(1 for e in self.effects if e * ref < 0)

    @property
    def robust(self) -> bool:
        """A conservative bar: ≥80% support and no sign reversals."""
        return self.support_rate >= 0.8 and self.sign_flips == 0

    # -- Attribution -------------------------------------------------------

    def fork_influence(self) -> dict[str, float]:
        """How much each fork moves the effect, as a share of total spread.

        Answers the question the specification curve exists for: *which
        arbitrary choice is driving the result?* A fork responsible for most
        of the variation is a finding about the analysis, not about nature.
        """
        if len(self.ran) < 2:
            return {}

        overall = statistics.pstdev(self.effects) if len(self.effects) > 1 else 0.0
        if overall == 0.0:
            return {k: 0.0 for k in self.ran[0].spec}

        influence: dict[str, float] = {}
        for fork in self.ran[0].spec:
            group_means = []
            for level in {r.spec[fork] for r in self.ran}:
                vals = [
                    r.effect for r in self.ran
                    if r.spec[fork] == level and r.effect is not None
                ]
                if vals:
                    group_means.append(statistics.fmean(vals))
            between = statistics.pstdev(group_means) if len(group_means) > 1 else 0.0
            influence[fork] = round(between / overall, 3)
        return dict(sorted(influence.items(), key=lambda kv: kv[1], reverse=True))

    # -- Rendering ---------------------------------------------------------

    def render(self) -> str:
        if not self.ran:
            return "Multiverse analysis: no specification executed successfully."

        lo, hi = self.effect_range or (0.0, 0.0)
        lines = [
            "## Multiverse (specification-curve) analysis",
            "",
            f"Specifications executed: {self.n_ran}/{self.n_total}",
            f"Supporting the hypothesis: {self.n_supporting} "
            f"({self.support_rate:.0%})",
            f"Fragility: {self.fragility:.2f}",
            f"Effect: median {self.median_effect:.4g}, range [{lo:.4g}, {hi:.4g}]",
            f"Sign reversals: {self.sign_flips}",
            "",
            f"**Verdict: {'ROBUST' if self.robust else 'FRAGILE'}** "
            + (
                "— the conclusion survives the analytic choices."
                if self.robust else
                "— the conclusion depends on arbitrary analytic choices."
            ),
        ]

        influence = self.fork_influence()
        if influence:
            lines += ["", "Variation attributable to each analytic fork:"]
            for fork, share in influence.items():
                bar = "█" * max(0, min(20, round(share * 20)))
                lines.append(f"  {fork:<18} {share:>5.2f} {bar}")
            top_fork, top_share = next(iter(influence.items()))
            if top_share > 0.5:
                lines += [
                    "",
                    f"⚠ '{top_fork}' alone accounts for most of the spread. "
                    "That is a fact about the analysis, not about the system "
                    "under study.",
                ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_ran": self.n_ran,
            "n_supporting": self.n_supporting,
            "support_rate": round(self.support_rate, 4),
            "fragility": round(self.fragility, 4),
            "median_effect": self.median_effect,
            "effect_range": self.effect_range,
            "sign_flips": self.sign_flips,
            "robust": self.robust,
            "fork_influence": self.fork_influence(),
            "specifications": [
                {
                    "spec": specification_id(r.spec),
                    "effect": r.effect,
                    "p_value": r.p_value,
                    "n": r.n,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Code generation for the sandboxed runner
# ---------------------------------------------------------------------------

#: Preamble injected above the generated analysis so every specification is
#: applied by the same, auditable code rather than re-implemented by the LLM
#: for each fork (which would confound analytic choice with code variation).
SPEC_PREAMBLE = '''
# --- Multiverse specification harness (injected) ---------------------------
import json as _json
import numpy as _np

SPEC = _json.loads(r"""__SPEC_JSON__""")


def _apply_outlier_policy(x, policy):
    x = _np.asarray(x, dtype=float)
    x = x[~_np.isnan(x)]
    if policy == "none" or x.size == 0:
        return x
    if policy == "iqr_1.5":
        q1, q3 = _np.percentile(x, [25, 75])
        iqr = q3 - q1
        return x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    if policy == "winsorize_5pct":
        lo, hi = _np.percentile(x, [5, 95])
        return _np.clip(x, lo, hi)
    if policy == "drop_3sd":
        mu, sd = _np.mean(x), _np.std(x)
        return x if sd == 0 else x[_np.abs(x - mu) <= 3 * sd]
    return x


def _apply_transform(x, transform):
    x = _np.asarray(x, dtype=float)
    if transform == "log":
        shift = 0.0 if x.min() > 0 else (abs(x.min()) + 1e-9)
        return _np.log(x + shift + 1e-9)
    if transform == "rank":
        order = x.argsort().argsort()
        return order.astype(float)
    return x


def _run_test(a, b, test):
    from scipy import stats as _st
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    if test == "welch_t":
        t, p = _st.ttest_ind(a, b, equal_var=False)
    elif test == "student_t":
        t, p = _st.ttest_ind(a, b, equal_var=True)
    elif test == "mann_whitney":
        t, p = _st.mannwhitneyu(a, b, alternative="two-sided")
    elif test == "permutation":
        res = _st.permutation_test(
            (a, b), lambda x, y, axis=0: _np.mean(x, axis=axis) - _np.mean(y, axis=axis),
            n_resamples=2000, alternative="two-sided", random_state=0,
        )
        t, p = res.statistic, res.pvalue
    else:
        t, p = _st.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def multiverse_analyse(group_a, group_b):
    """Apply SPEC to two groups and emit the harness result line."""
    a = _apply_transform(_apply_outlier_policy(group_a, SPEC["outlier_policy"]),
                         SPEC["transform"])
    b = _apply_transform(_apply_outlier_policy(group_b, SPEC["outlier_policy"]),
                         SPEC["transform"])
    _stat, p = _run_test(a, b, SPEC["test"])
    pooled = _np.sqrt((_np.var(a, ddof=1) + _np.var(b, ddof=1)) / 2) if (
        a.size > 1 and b.size > 1) else 0.0
    effect = float((_np.mean(a) - _np.mean(b)) / pooled) if pooled > 0 else 0.0
    if SPEC["covariate_adjust"] == "adjusted":
        effect *= 0.9   # placeholder shrinkage when no covariates are supplied
    print("SPEC_RESULT:" + _json.dumps({
        "effect": effect, "p_value": p, "n": int(a.size + b.size),
    }))
    return effect, p
# --- end harness -----------------------------------------------------------
'''


def build_specification_code(base_code: str, spec: dict[str, str]) -> str:
    """Inject one specification into the generated analysis script."""
    import json

    preamble = SPEC_PREAMBLE.replace("__SPEC_JSON__", json.dumps(spec))
    return preamble + "\n" + base_code


def parse_specification_result(stdout: str, spec: dict[str, str]) -> SpecificationResult:
    """Read the ``SPEC_RESULT:`` line emitted by the harness."""
    import json

    marker = "SPEC_RESULT:"
    if marker not in (stdout or ""):
        return SpecificationResult(spec=spec, error="no SPEC_RESULT line emitted")

    tail = stdout.rsplit(marker, 1)[1].strip().splitlines()[0]
    try:
        payload = json.loads(tail)
    except json.JSONDecodeError as exc:
        return SpecificationResult(spec=spec, error=f"unparseable result: {exc}")

    effect = payload.get("effect")
    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return SpecificationResult(spec=spec, error="effect is NaN")

    p_value = payload.get("p_value")
    if isinstance(p_value, float) and math.isnan(p_value):
        p_value = None

    return SpecificationResult(
        spec=spec,
        effect=float(effect),
        p_value=None if p_value is None else float(p_value),
        n=payload.get("n"),
    )


__all__ = [
    "DEFAULT_FORKS",
    "MultiverseReport",
    "SPEC_PREAMBLE",
    "SpecificationResult",
    "build_specification_code",
    "enumerate_specifications",
    "parse_specification_result",
    "specification_id",
]
