"""
agents/supervisor.py — SupervisorAgent for task queue orchestration.

Responsible for:
- Managing a priority task queue
- Registering and dispatching tasks to specialized agents
- Tracking task execution history
"""

from __future__ import annotations

import heapq
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class Task:
    """Represents an async task in the worker queue"""

    def __init__(self, agent_name: str, action: str, params: dict, priority: int = 5):
        self.id = str(uuid.uuid4())[:8]
        self.agent_name = agent_name
        self.action = action
        self.params = params
        self.priority = priority  # Lower number = higher priority
        self.created_at = datetime.now()
        self.completed_at = None
        self.result = None

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


class SupervisorAgent:
    """Orchestrates all specialized agents and manages task queue"""

    def __init__(self):
        self.name = "Supervisor"
        self.task_queue = []
        self.task_history = []
        self.agent_registry = {}
        self.iteration = 0

    def register_agent(self, agent):
        """Register a specialized agent"""
        self.agent_registry[agent.name] = agent

    async def execute_task_queue(self, max_iterations: int = 3):
        """Execute queued tasks"""
        for _ in range(max_iterations):
            if not self.task_queue:
                break

            task = heapq.heappop(self.task_queue)
            agent = self.agent_registry.get(task.agent_name)

            if agent:
                action_method = getattr(agent, task.action, None)
                if action_method and callable(action_method):
                    try:
                        task.result = await action_method(**task.params)
                    except Exception as e:
                        logger.error(f"Task {task.action} on {agent.name} failed: {e}")
                        task.result = e
                else:
                    logger.warning(f"Action {task.action} not found on agent {agent.name}")

                task.completed_at = datetime.now()
                self.task_history.append(task)

            self.iteration += 1

    def queue_task(self, agent_name: str, action: str, params: dict, priority: int = 5):
        """Add task to queue"""
        task = Task(agent_name, action, params, priority)
        heapq.heappush(self.task_queue, task)
        return task.id

    def get_task_stats(self) -> dict:
        """Get statistics on task execution"""
        return {
            "total_tasks_completed": len(self.task_history),
            "pending_tasks": len(self.task_queue),
            "iterations_completed": self.iteration,
            "agents_registered": list(self.agent_registry.keys())
        }


__all__ = ["SupervisorAgent", "Task"]
