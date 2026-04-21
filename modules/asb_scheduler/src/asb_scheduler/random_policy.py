"""Random scheduling policy for build_execution_queue."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def build_random_queue(
    items: Sequence[T],
    repetitions: int,
    seed: int | None = None,
) -> list[tuple[T, int]]:
    """Build a randomly shuffled execution queue.

    Same as sequential ordering, but shuffled using the given seed.
    """
    queue: list[tuple[T, int]] = [(x, r) for x in items for r in range(repetitions)]
    random.Random(seed).shuffle(queue)
    return queue
