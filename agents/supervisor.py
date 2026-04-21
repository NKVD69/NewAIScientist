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
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


class Task:
    """Represents an async task in the worker queue"""
    
    def __init__(self, agent_name: str, action: str, params: Dict, priority: int = 5):
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
                if task.action == "generate":
                    task.result = await agent.generate_initial_hypotheses(**task.params)
                elif task.action == "review":
                    task.result = await agent.review_hypothesis(**task.params)
                elif task.action == "tournament":
                    task.result = await agent.conduct_tournament_match(**task.params)
                elif task.action == "compute_proximity":
                    task.result = await agent.compute_proximity(**task.params)
                elif task.action == "evolve":
                    task.result = await agent.evolve_hypothesis(**task.params)
                elif task.action == "meta_review":
                    task.result = await agent.generate_meta_review(**task.params)
                elif task.action == "search_literature":
                    task.result = await agent.search_literature(**task.params)
                elif task.action == "experiment":
                    task.result = await agent.run_experiment(**task.params)
                
                task.completed_at = datetime.now()
                self.task_history.append(task)
            
            self.iteration += 1
    
    def queue_task(self, agent_name: str, action: str, params: Dict, priority: int = 5):
        """Add task to queue"""
        task = Task(agent_name, action, params, priority)
        heapq.heappush(self.task_queue, task)
        return task.id
    
    def get_task_stats(self) -> Dict:
        """Get statistics on task execution"""
        return {
            "total_tasks_completed": len(self.task_history),
            "pending_tasks": len(self.task_queue),
            "iterations_completed": self.iteration,
            "agents_registered": list(self.agent_registry.keys())
        }


__all__ = ["SupervisorAgent", "Task"]
