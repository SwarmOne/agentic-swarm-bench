"""Load and filter benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

TASKS_FILE = Path(__file__).parent / "tasks.json"

TIERS = ["trivial", "easy", "medium", "hard", "expert"]

TIER_RANGES = {
    "trivial": (1, 10),
    "easy": (11, 25),
    "medium": (26, 50),
    "hard": (51, 75),
    "expert": (76, 100),
}


def load_all_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return json.load(f)


def parse_task_range(spec: str) -> tuple[int, int]:
    """Parse a range like 'p1-p25', 'P51-P75', '1-50', or 'p10'.

    Returns (start, end) inclusive.
    """
    spec = spec.strip().lower()

    if spec in TIERS:
        return TIER_RANGES[spec]

    spec = spec.replace("p", "")
    if "-" in spec:
        parts = spec.split("-", 1)
        return int(parts[0]), int(parts[1])

    n = int(spec)
    return n, n


def _task_number(task: dict) -> int:
    return int(task["id"].lstrip("Pp"))


def filter_tasks(
    tasks: list[dict],
    task_range: Optional[str] = None,
    tier: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """Filter tasks by range, tier, or tags."""
    result = tasks

    if task_range:
        start, end = parse_task_range(task_range)
        result = [t for t in result if start <= _task_number(t) <= end]

    if tier:
        result = [t for t in result if t["tier"] == tier]

    if tags:
        tag_set = set(tags)
        result = [t for t in result if tag_set.intersection(t.get("tags", []))]

    return result


def get_tasks(
    task_range: Optional[str] = None,
    tier: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """Load and filter tasks in one call."""
    all_tasks = load_all_tasks()
    return filter_tasks(all_tasks, task_range=task_range, tier=tier, tags=tags)
