"""Execution schedule for scenario replay.

Controls how tasks from a scenario are ordered and rate-limited during replay:
  - repetitions: how many times each task runs
  - max_concurrent: maximum tasks executing at the same time
  - policy: ordering strategy (round_robin, sequential, random)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentic_swarm_bench.scenarios.registry import Task

VALID_POLICIES = ("round_robin", "sequential", "random")


@dataclass
class Schedule:
    repetitions: int = 1
    max_concurrent: int = 10
    policy: str = "round_robin"

    def __post_init__(self):
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {self.policy!r}")


def build_execution_queue(
    tasks: list[Task],
    schedule: Schedule,
    *,
    seed: int | None = None,
) -> list[tuple[Task, int]]:
    """Generate the ordered list of (task, execution_index) from a schedule.

    Returns a flat list where each element is a (task, execution_index) tuple.
    The execution_index is the repetition number for that task (0-based).

    Policies:
      round_robin: T1_0,T2_0,T3_0,T1_1,T2_1,T3_1,...
      sequential:  T1_0,T1_1,T1_2,T2_0,T2_1,T2_2,...
      random:      shuffled
    """
    if not tasks:
        return []

    if schedule.policy == "sequential":
        queue = [(t, rep) for t in tasks for rep in range(schedule.repetitions)]
    elif schedule.policy == "round_robin":
        queue = [(t, rep) for rep in range(schedule.repetitions) for t in tasks]
    elif schedule.policy == "random":
        queue = [(t, rep) for t in tasks for rep in range(schedule.repetitions)]
        rng = random.Random(seed)
        rng.shuffle(queue)
    else:
        raise ValueError(f"Unknown policy: {schedule.policy!r}")

    return queue
