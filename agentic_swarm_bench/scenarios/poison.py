"""Prefix-cache poisoning for replay scenarios.

Provides compute_scenario_lcp() and poison_task_execution() used by
the replay engine. Cache-defeat implementations are loaded from
extension modules when available; otherwise these return no-op values
(LCP=0, task unchanged) which is correct for allwarm cache mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_swarm_bench.scenarios.registry import Task


# --- PRIVATE ---
try:
    from asb_cache_defeat.live_poison import (
        compute_scenario_lcp as _real_compute_scenario_lcp,
    )
    from asb_cache_defeat.live_poison import (
        poison_task_execution as _real_poison_task_execution,
    )

    _HAS_CACHE_DEFEAT = True
except ImportError:
    _HAS_CACHE_DEFEAT = False
# --- /PRIVATE ---


def compute_scenario_lcp(tasks: list[Task]) -> int:
    # --- PRIVATE ---
    if _HAS_CACHE_DEFEAT:
        return _real_compute_scenario_lcp(tasks)
    # --- /PRIVATE ---
    return 0


def poison_task_execution(
    task: Task,
    lcp_len: int,
    execution_index: int,
) -> Task:
    # --- PRIVATE ---
    if _HAS_CACHE_DEFEAT:
        return _real_poison_task_execution(task, lcp_len, execution_index)
    # --- /PRIVATE ---
    return task
