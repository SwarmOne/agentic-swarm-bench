"""Interleaved random scheduling -- per-request dispatch across task executions.

Provides the pre-computed ordering function and the dependency-aware
work-queue dispatcher.
"""

from __future__ import annotations

import asyncio
import collections
import random
from typing import Awaitable, Callable, Sequence, TypeVar

R = TypeVar("R")


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
