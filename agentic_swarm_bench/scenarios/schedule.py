"""Schedule-task model and work-queue dispatcher.

Mike's mental model (see docs/SCHEDULING.md):

    schedule-task = (task, execution_index)
        one task run once; each has r_1..r_n underlying request entries.

    With T distinct tasks and R repetitions, there are T*R schedule-tasks
    in total. Given J concurrency slots, we:

      1. Pre-generate a single ordered list L of all T*R schedule-tasks
         using one of four policies:
            - sequential:          t1r1, t1r2, ..., t1rR, t2r1, ..., t2rR, ...
            - round_robin:         t1r1, t2r1, ..., tTr1, t1r2, ..., tTrR
            - random:              shuffle(sequential)
            - interleaved_random:  shuffle individual requests across tasks

      2. Spin up exactly J long-lived workers. Each worker pulls the
         next item off the head of L, runs it, then pulls the next.
         No worker ever waits for the others. When L is empty, workers
         exit. This is a rolling pool, not a lockstep batch.

The ``Schedule`` dataclass bundles (R, J, policy, seed). The
``build_execution_queue`` function produces L. The ``run_work_queue``
coroutine is the literal pool-of-J dispatcher.
"""

from __future__ import annotations

import asyncio
import collections
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence, TypeVar

VALID_POLICIES = ("round_robin", "sequential", "random", "interleaved_random")


@dataclass
class Schedule:
    """Concurrency + ordering controls for a benchmark run.

    Attributes:
        repetitions:    R, how many times each task runs.
        max_concurrent: J, pool size of parallel workers.
        policy:         How L is ordered before dispatch.
        seed:           RNG seed for policy="random" (None = non-reproducible).
    """

    repetitions: int = 1
    max_concurrent: int = 10
    policy: str = "round_robin"
    seed: int | None = None

    def __post_init__(self):
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {self.policy!r}")


T = TypeVar("T")
R = TypeVar("R")


def build_execution_queue(
    items: Sequence[T],
    schedule: Schedule,
    *,
    seed: int | None = None,
) -> list[tuple[T, int]]:
    """Build the ordered pending list L of (item, execution_index) schedule-tasks.

    ``items`` is typically a list of ``Task`` (replay) or task dicts (agent).
    ``execution_index`` is the 0-based repetition number for that item.

    Policies:
      sequential:  [(t1,0),(t1,1)..(t1,R-1),(t2,0),(t2,1)..]
      round_robin: [(t1,0),(t2,0)..(tN,0),(t1,1),(t2,1)..]
      random:      shuffle(sequential)

    Seed precedence: explicit ``seed=`` argument wins over ``schedule.seed``.
    When the effective seed is None, random.Random(None) uses system entropy
    so every call produces a different order. Pass a seed to reproduce runs.
    """
    if not items:
        return []

    reps = schedule.repetitions

    if schedule.policy == "sequential":
        return [(x, r) for x in items for r in range(reps)]

    if schedule.policy == "round_robin":
        return [(x, r) for r in range(reps) for x in items]

    if schedule.policy == "random":
        queue: list[tuple[T, int]] = [(x, r) for x in items for r in range(reps)]
        effective_seed = seed if seed is not None else schedule.seed
        random.Random(effective_seed).shuffle(queue)
        return queue

    if schedule.policy == "interleaved_random":
        return [(x, r) for x in items for r in range(reps)]

    raise ValueError(f"Unknown policy: {schedule.policy!r}")


async def run_work_queue(
    queue: Sequence[T],
    worker: Callable[[T, int], Awaitable[R]],
    max_concurrent: int,
) -> list[R]:
    """Dispatch ``queue`` via a rolling pool of ``max_concurrent`` workers.

    Each of the J workers loops:
        while pending:
            item = pending.popleft()
            result = await worker(item, slot_id)

    Workers never wait for each other. As soon as a worker finishes its
    current item, it pulls the next head of pending. If J exceeds the
    queue length, excess workers exit immediately without running anything.

    Safe under asyncio because ``deque.popleft`` is atomic between awaits:
    each worker reads/pops the queue without yielding, so two workers can
    never pop the same item.

    Returns results in input order: ``results[i]`` is the output of
    ``worker(queue[i], ...)``. None entries indicate skipped or failed
    items (worker returning None).
    """
    n = len(queue)
    if n == 0:
        return []

    pending: collections.deque[tuple[int, T]] = collections.deque(enumerate(queue))
    results: list[R | None] = [None] * n

    async def worker_loop(slot_id: int) -> None:
        while pending:
            idx, item = pending.popleft()
            results[idx] = await worker(item, slot_id)

    j = max(1, min(max_concurrent, n))
    await asyncio.gather(*[worker_loop(i) for i in range(j)])
    return results  # type: ignore[return-value]


def build_interleaved_order(
    entry_counts: Sequence[int],
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Pre-compute a random interleaving of individual requests across task executions.

    Each task execution ``i`` has ``entry_counts[i]`` sequential requests.
    Returns a list of ``(task_exec_idx, entry_idx)`` tuples where within-task
    ordering is preserved but the interleaving across tasks is random.

    Algorithm: maintain a cursor per task execution tracking the next entry
    to emit. At each step, randomly pick a task execution that still has
    entries remaining and emit its next entry. This produces a random
    topological sort of the dependency chains.

    Invariant: for any task execution ``i``, entry ``n`` always appears
    before entry ``n+1`` in the output. This guarantees no deadlocks in
    the dependency-aware dispatcher, because a request's predecessor is
    always earlier in the sequence.
    """
    rng = random.Random(seed)
    cursors = list(range(len(entry_counts)))
    next_entry = [0] * len(entry_counts)
    remaining = list(entry_counts)
    total = sum(entry_counts)

    order: list[tuple[int, int]] = []
    for _ in range(total):
        active = [i for i in cursors if remaining[i] > 0]
        pick = rng.choice(active)
        order.append((pick, next_entry[pick]))
        next_entry[pick] += 1
        remaining[pick] -= 1

    return order


async def run_interleaved_work_queue(
    order: Sequence[tuple[int, int]],
    worker: Callable[[int, int, int], Awaitable[R]],
    max_concurrent: int,
) -> list[R | None]:
    """Dispatch an interleaved request order via a pool of J workers.

    ``order`` is the pre-computed sequence from ``build_interleaved_order``:
    each item is ``(task_exec_idx, entry_idx)``.

    ``worker(task_exec_idx, entry_idx, slot_id)`` processes one request.

    Workers pull from the head of the order. Before executing, a worker
    waits for the request's predecessor in the same task execution to
    complete (entry_idx - 1). This enforces within-task serialization
    while allowing cross-task parallelism.

    Since ``build_interleaved_order`` guarantees entry N appears before
    entry N+1 for any task execution, the predecessor is always earlier
    in the sequence and will complete before (or concurrently with)
    the dependent item. Deadlocks are impossible.
    """
    n = len(order)
    if n == 0:
        return []

    completion: dict[tuple[int, int], asyncio.Event] = {}
    for task_exec_idx, entry_idx in order:
        completion[(task_exec_idx, entry_idx)] = asyncio.Event()

    pending: collections.deque[tuple[int, tuple[int, int]]] = collections.deque(
        enumerate(order)
    )
    results: list[R | None] = [None] * n

    async def worker_loop(slot_id: int) -> None:
        while pending:
            idx, (task_exec_idx, entry_idx) = pending.popleft()
            if entry_idx > 0:
                await completion[(task_exec_idx, entry_idx - 1)].wait()
            results[idx] = await worker(task_exec_idx, entry_idx, slot_id)
            completion[(task_exec_idx, entry_idx)].set()

    j = max(1, min(max_concurrent, n))
    await asyncio.gather(*[worker_loop(i) for i in range(j)])
    return results
