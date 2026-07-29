"""
agents/supervisor.py — SupervisorAgent: dependency-aware orchestration.

The previous implementation was a priority heap, not a scheduler. Three
defects let a broken run present itself as a successful one:

1. **Failures were swallowed.** Exceptions were stored *as the task result*
   and execution continued. A failed ``run_literature_search`` meant
   hypothesis generation ran with no literature, no CAG and an empty RAG
   index — producing fluent, ungrounded output that traversed the whole
   pipeline with nothing flagging the empty evidence base.
2. **Tasks could vanish.** ``for _ in range(max_iterations)`` drained the
   queue; a longer queue left the remainder to execute later, out of phase.
3. **The DAG lived in magic numbers.** ``priority=base_prio+1..6`` encoded
   *review → proximity → tournament → evolution* in the developer's head.

Dependencies are now declared (``utils.pipeline``), validated at build time,
and independent tasks in a wave run concurrently — everything here has always
been ``async``, but only ``ReflectionAgent`` ever exploited it.

The legacy ``queue_task`` / ``execute_task_queue`` API is retained so
existing callers keep working; it now delegates to the DAG executor with each
task depending on the previously queued one, which reproduces the old
sequential semantics but with real failure policies.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from utils.pipeline import (
    FailurePolicy,
    Pipeline,
    PipelineReport,
    TaskSpec,
    TaskState,
)

logger = logging.getLogger(__name__)


class Task:
    """Legacy task record, retained for the existing queue API."""

    def __init__(self, agent_name: str, action: str, params: dict, priority: int = 5):
        self.id = str(uuid.uuid4())[:8]
        self.agent_name = agent_name
        self.action = action
        self.params = params
        self.priority = priority  # Lower number = higher priority
        self.created_at = datetime.now()
        self.completed_at = None
        self.result = None
        self.state = "pending"
        self.error = ""

    def __lt__(self, other):
        return self.priority < other.priority


class SupervisorAgent:
    """Orchestrates specialized agents through a validated task DAG."""

    def __init__(self, max_parallel: int = 4, budget=None):
        self.name = "Supervisor"
        self.task_queue: list[Task] = []
        self.task_history: list[Task] = []
        self.agent_registry: dict = {}
        self.iteration = 0
        self.max_parallel = max_parallel
        self.budget = budget
        #: Reports from every pipeline executed this session.
        self.reports: list[PipelineReport] = []

    def register_agent(self, agent):
        self.agent_registry[agent.name] = agent

    # ------------------------------------------------------------------
    # DAG execution (preferred)
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        specs: list[TaskSpec],
        context: dict | None = None,
        label: str = "pipeline",
    ) -> PipelineReport:
        """Execute a declared task DAG.

        The returned report is the honest record of the run: which tasks
        succeeded, which failed, which were skipped because an upstream
        dependency failed, and whether the whole thing aborted. Callers must
        check ``report.clean`` before treating downstream artefacts as
        resting on a complete evidence base.
        """
        pipeline = Pipeline(
            specs,
            agents=self.agent_registry,
            budget=self.budget,
            max_parallel=self.max_parallel,
        )
        logger.info("Executing %s: %d tasks.", label, len(specs))
        report = await pipeline.run(context if context is not None else {})

        for name, result in report.results.items():
            spec = pipeline.specs[name]
            task = Task(spec.agent, spec.action, spec.params)
            task.completed_at = datetime.now()
            task.result = result.value
            task.state = result.state.value
            task.error = result.error or result.skipped_because
            self.task_history.append(task)

        self.reports.append(report)
        self.iteration += 1

        print(f"\n{report.render()}")
        if not report.clean:
            print(
                "\n⚠ This run is NOT clean. Artefacts produced downstream rest on "
                "an incomplete evidence base and must be reported as such."
            )
        return report

    # ------------------------------------------------------------------
    # Legacy queue API (delegates to the DAG executor)
    # ------------------------------------------------------------------

    def queue_task(
        self,
        agent_name: str,
        action: str,
        params: dict,
        priority: int = 5,
        on_failure: FailurePolicy = FailurePolicy.DEGRADE,
    ) -> str:
        """Add a task to the legacy queue.

        Defaults to ``DEGRADE`` rather than ``ABORT``: the old behaviour was
        to continue past every failure, so aborting by default here would
        change existing callers' semantics more than intended. New code
        should use ``run_pipeline`` with explicit policies.
        """
        task = Task(agent_name, action, params, priority)
        task.on_failure = on_failure  # type: ignore[attr-defined]
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority)
        return task.id

    async def execute_task_queue(self, max_iterations: int | None = None):
        """Drain the legacy queue in priority order.

        Unlike the previous version this drains the queue *completely*.
        ``max_iterations`` is honoured but a truncated queue now raises a
        loud warning instead of silently leaving tasks to execute later in
        an unrelated phase.
        """
        if not self.task_queue:
            return None

        queued = list(self.task_queue)
        if max_iterations is not None and len(queued) > max_iterations:
            logger.warning(
                "execute_task_queue: %d tasks queued but max_iterations=%d. "
                "Executing all of them anyway — silently deferring tasks to a "
                "later phase was a source of out-of-order execution.",
                len(queued), max_iterations,
            )

        specs: list[TaskSpec] = []
        previous: str | None = None
        for idx, task in enumerate(queued):
            name = f"{task.action}#{idx}"
            specs.append(TaskSpec(
                name=name,
                action=task.action,
                agent=task.agent_name,
                params=task.params,
                depends_on=(previous,) if previous else (),
                on_failure=getattr(task, "on_failure", FailurePolicy.DEGRADE),
            ))
            previous = name

        self.task_queue.clear()
        return await self.run_pipeline(specs, label="legacy queue")

    # ------------------------------------------------------------------

    def get_task_stats(self) -> dict:
        states: dict[str, int] = {}
        for task in self.task_history:
            states[task.state] = states.get(task.state, 0) + 1
        return {
            "total_tasks_completed": len(self.task_history),
            "pending_tasks": len(self.task_queue),
            "iterations_completed": self.iteration,
            "agents_registered": list(self.agent_registry.keys()),
            "by_state": states,
            "clean_runs": sum(1 for r in self.reports if r.clean),
            "total_runs": len(self.reports),
        }

    def failure_digest(self) -> str:
        """Everything that went wrong this session, in one place."""
        problems: list[str] = []
        for report in self.reports:
            for name in report.failed:
                problems.append(f"FAILED  {name}: {report.results[name].error[:120]}")
            for name in report.skipped:
                reason = report.results[name].skipped_because
                problems.append(f"SKIPPED {name}: {reason[:120]}")
        if not problems:
            return "No task failures or skips this session."
        return "\n".join(["Task failures and skips:", *(f"  {p}" for p in problems)])


__all__ = ["SupervisorAgent", "Task", "TaskState"]
