"""
utils/pipeline.py — Dependency-aware orchestration with explicit failure policies.

Replaces the integer-priority task queue, which had three defects that let a
broken run masquerade as a successful one.

**1. Failures were swallowed.** ``execute_task_queue`` caught every exception,
stored it *as the task result*, and moved on::

    except Exception as e:
        logger.error(f"Task {task.action} failed: {e}")
        task.result = e          # ← the exception becomes the result

So if ``run_literature_search`` failed (network, arXiv down, NCBI quota),
hypothesis generation still ran — with no literature, no CAG context and an
empty RAG index. It produced fluent, ungrounded, hallucinated hypotheses that
then traversed tournament, experiment and write-up with nothing flagging that
the evidence base was empty.

**2. Tasks could be dropped silently.** The queue was drained by
``for _ in range(max_iterations)``. Queue longer than the bound? The remainder
stayed queued and executed later, in a different phase, out of context.

**3. The dependency graph lived in magic numbers.** ``priority=base_prio+1..6``
encoded *review → proximity → tournament → evolution* in the developer's head.
Nothing enforced it.

Here the graph is declared, so it can be validated, parallelised and reported
on. Independent tasks in the same wave run concurrently — everything is already
``async`` and the codebase only parallelised inside a single agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailurePolicy(Enum):
    """What happens to the run when a task raises."""

    #: Stop the pipeline. For tasks whose output everything downstream assumes.
    ABORT = "abort"
    #: Record the failure, skip dependents, let independent branches continue.
    DEGRADE = "degrade"
    #: Retry with backoff, then fall back to ABORT.
    RETRY = "retry"
    #: Failure is expected and carries no information. Use sparingly.
    IGNORE = "ignore"


class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: Never ran: an upstream dependency failed.
    SKIPPED = "skipped"


@dataclass
class TaskSpec:
    """A node in the pipeline DAG."""

    name: str
    action: str
    agent: str = "Orchestrator"
    params: dict = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    on_failure: FailurePolicy = FailurePolicy.ABORT
    max_retries: int = 2
    retry_base_delay: float = 1.0
    #: Skip the task entirely when this returns False. Receives the context.
    condition: Callable[[dict], bool] | None = None
    #: Rough token cost estimate, used for budget pre-flight checks.
    estimated_tokens: int = 0
    description: str = ""


@dataclass
class TaskResult:
    name: str
    state: TaskState = TaskState.PENDING
    value: Any = None
    error: str = ""
    attempts: int = 0
    duration_s: float = 0.0
    skipped_because: str = ""

    @property
    def ok(self) -> bool:
        return self.state is TaskState.SUCCEEDED


@dataclass
class PipelineReport:
    """Outcome of one pipeline execution — the honest record of what happened."""

    results: dict[str, TaskResult] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""
    waves: list[list[str]] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def succeeded(self) -> list[str]:
        return [n for n, r in self.results.items() if r.state is TaskState.SUCCEEDED]

    @property
    def failed(self) -> list[str]:
        return [n for n, r in self.results.items() if r.state is TaskState.FAILED]

    @property
    def skipped(self) -> list[str]:
        return [n for n, r in self.results.items() if r.state is TaskState.SKIPPED]

    @property
    def clean(self) -> bool:
        """True iff every task ran and succeeded.

        The distinction that matters downstream: a report that is not ``clean``
        means the artefacts produced by this run rest on an incomplete
        evidence base, and any manuscript derived from it must say so.
        """
        return not self.aborted and not self.failed and not self.skipped

    def value(self, task_name: str, default: Any = None) -> Any:
        result = self.results.get(task_name)
        return result.value if result and result.ok else default

    def summary(self) -> str:
        parts = [
            f"{len(self.succeeded)} succeeded",
            f"{len(self.failed)} failed",
            f"{len(self.skipped)} skipped",
            f"{self.duration_s:.1f}s",
            f"{len(self.waves)} waves",
        ]
        line = " · ".join(parts)
        if self.aborted:
            line += f" — ABORTED: {self.abort_reason}"
        return line

    def render(self) -> str:
        """Full per-task report. Printed at the end of every run."""
        icons = {
            TaskState.SUCCEEDED: "✅", TaskState.FAILED: "❌",
            TaskState.SKIPPED: "⬜", TaskState.PENDING: "…",
            TaskState.RUNNING: "▶",
        }
        lines = ["Pipeline report", "─" * 60]
        for wave_idx, wave in enumerate(self.waves, 1):
            lines.append(f"Wave {wave_idx}" + (" (parallel)" if len(wave) > 1 else ""))
            for name in wave:
                r = self.results.get(name)
                if r is None:
                    continue
                icon = icons.get(r.state, "•")
                detail = f" ({r.duration_s:.1f}s)" if r.duration_s else ""
                if r.attempts > 1:
                    detail += f" after {r.attempts} attempts"
                lines.append(f"  {icon} {name}{detail}")
                if r.error:
                    lines.append(f"      ↳ {r.error[:160]}")
                if r.skipped_because:
                    lines.append(f"      ↳ {r.skipped_because}")
        lines.append("─" * 60)
        lines.append(self.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """Raised for a malformed DAG — a programming error, not a runtime one."""


def validate(specs: list[TaskSpec]) -> None:
    """Reject duplicate names, unknown dependencies and cycles.

    Fails at construction rather than mid-run, which is the whole point of
    declaring the graph instead of encoding it in priority integers.
    """
    names = [s.name for s in specs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise PipelineError(f"duplicate task names: {sorted(duplicates)}")

    known = set(names)
    for spec in specs:
        unknown = set(spec.depends_on) - known
        if unknown:
            raise PipelineError(
                f"task '{spec.name}' depends on unknown task(s): {sorted(unknown)}"
            )

    # Cycle detection by iterative peeling.
    pending = {s.name: set(s.depends_on) for s in specs}
    while pending:
        ready = [n for n, deps in pending.items() if not deps]
        if not ready:
            raise PipelineError(
                f"dependency cycle among: {sorted(pending)}"
            )
        for name in ready:
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)


def topological_waves(specs: list[TaskSpec]) -> list[list[str]]:
    """Group tasks into waves of mutually independent tasks.

    Everything in a wave can run concurrently. The old executor ran strictly
    sequentially even though every agent method is ``async``.
    """
    validate(specs)
    pending = {s.name: set(s.depends_on) for s in specs}
    waves: list[list[str]] = []
    while pending:
        ready = sorted(n for n, deps in pending.items() if not deps)
        waves.append(ready)
        for name in ready:
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)
    return waves


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Pipeline:
    """Executes a task DAG against a registry of agents."""

    def __init__(
        self,
        specs: list[TaskSpec],
        agents: dict[str, Any],
        budget: Any = None,
        max_parallel: int = 4,
    ):
        validate(specs)
        self.specs = {s.name: s for s in specs}
        self.order = [s.name for s in specs]
        self.agents = agents
        #: Optional ``utils.budget.BudgetTracker``. Checked before each wave.
        self.budget = budget
        self._semaphore = asyncio.Semaphore(max(1, max_parallel))

    # ------------------------------------------------------------------

    async def run(self, context: dict | None = None) -> PipelineReport:
        """Execute every task, honouring dependencies and failure policies."""
        context = context if context is not None else {}
        report = PipelineReport()
        report.waves = topological_waves(list(self.specs.values()))
        report.results = {n: TaskResult(name=n) for n in self.specs}

        started = time.monotonic()

        for wave in report.waves:
            if report.aborted:
                self._mark_remaining_skipped(report, "pipeline aborted upstream")
                break

            if self.budget is not None and self.budget.exhausted:
                report.aborted = True
                report.abort_reason = f"budget exhausted: {self.budget.summary()}"
                self._mark_remaining_skipped(report, "budget exhausted")
                break

            runnable = [
                name for name in wave
                if self._should_run(name, report, context)
            ]

            if runnable:
                outcomes = await asyncio.gather(*[
                    self._run_one(self.specs[name], context) for name in runnable
                ])
                for result in outcomes:
                    report.results[result.name] = result

            for name in runnable:
                result = report.results[name]
                if result.state is TaskState.FAILED:
                    policy = self.specs[name].on_failure
                    if policy in (FailurePolicy.ABORT, FailurePolicy.RETRY):
                        report.aborted = True
                        report.abort_reason = (
                            f"task '{name}' failed under policy {policy.value}: "
                            f"{result.error[:200]}"
                        )
                        logger.error(
                            "Pipeline aborted — %s. Downstream tasks will NOT run "
                            "on a corrupted state.", report.abort_reason,
                        )
                        break

        report.duration_s = time.monotonic() - started
        self._mark_remaining_skipped(report, "not reached")
        return report

    # ------------------------------------------------------------------

    def _should_run(self, name: str, report: PipelineReport, context: dict) -> bool:
        spec = self.specs[name]
        result = report.results[name]

        for dep in spec.depends_on:
            dep_result = report.results.get(dep)
            if dep_result is None or not dep_result.ok:
                dep_state = dep_result.state.value if dep_result else "missing"
                result.state = TaskState.SKIPPED
                result.skipped_because = (
                    f"dependency '{dep}' did not succeed ({dep_state})"
                )
                logger.warning(
                    "Skipping '%s': %s. This is deliberate — running it would "
                    "produce output that looks valid but rests on missing input.",
                    name, result.skipped_because,
                )
                return False

        if spec.condition is not None:
            try:
                if not spec.condition(context):
                    result.state = TaskState.SKIPPED
                    result.skipped_because = "condition not met"
                    return False
            except Exception as exc:  # noqa: BLE001
                result.state = TaskState.SKIPPED
                result.skipped_because = f"condition raised: {exc}"
                return False

        return True

    async def _run_one(self, spec: TaskSpec, context: dict) -> TaskResult:
        result = TaskResult(name=spec.name, state=TaskState.RUNNING)
        agent = self.agents.get(spec.agent)

        if agent is None:
            result.state = TaskState.FAILED
            result.error = f"agent '{spec.agent}' is not registered"
            return result

        method = getattr(agent, spec.action, None)
        if not callable(method):
            result.state = TaskState.FAILED
            result.error = f"agent '{spec.agent}' has no action '{spec.action}'"
            return result

        attempts = 1 + (spec.max_retries if spec.on_failure is FailurePolicy.RETRY else 0)
        started = time.monotonic()

        for attempt in range(1, attempts + 1):
            result.attempts = attempt
            try:
                async with self._semaphore:
                    value = method(**spec.params)
                    if asyncio.iscoroutine(value):
                        value = await value
                result.value = value
                result.state = TaskState.SUCCEEDED
                result.duration_s = time.monotonic() - started
                context[spec.name] = value
                return result
            except Exception as exc:  # noqa: BLE001
                result.error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Task '%s' failed (attempt %d/%d): %s",
                    spec.name, attempt, attempts, result.error,
                )
                if attempt < attempts:
                    await asyncio.sleep(spec.retry_base_delay * (2 ** (attempt - 1)))

        result.duration_s = time.monotonic() - started
        if spec.on_failure is FailurePolicy.IGNORE:
            result.state = TaskState.SUCCEEDED
            logger.info("Task '%s' failed but policy is IGNORE.", spec.name)
        else:
            result.state = TaskState.FAILED
        return result

    @staticmethod
    def _mark_remaining_skipped(report: PipelineReport, reason: str) -> None:
        for result in report.results.values():
            if result.state in (TaskState.PENDING, TaskState.RUNNING):
                result.state = TaskState.SKIPPED
                result.skipped_because = result.skipped_because or reason


__all__ = [
    "FailurePolicy",
    "Pipeline",
    "PipelineError",
    "PipelineReport",
    "TaskResult",
    "TaskSpec",
    "TaskState",
    "topological_waves",
    "validate",
]
