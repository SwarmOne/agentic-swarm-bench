"""ASB Scheduler module -- random and interleaved scheduling policies.

Proprietary module for AgenticSwarmBench. Provides:
  - random policy for build_execution_queue
  - interleaved_random policy with dependency-aware dispatch
"""

from __future__ import annotations

from asb_scheduler.interleaved import build_interleaved_order, run_interleaved_work_queue
from asb_scheduler.random_policy import build_random_queue

__all__ = [
    "build_interleaved_order",
    "build_random_queue",
    "run_interleaved_work_queue",
]

EXTRA_POLICIES = ("random", "interleaved_random")


class SchedulerModule:
    """Entry-point class for the ASB plugin system."""

    name = "scheduler"

    def register_cli_flags(self, group):
        pass

    def apply(self, context: dict):
        pass
