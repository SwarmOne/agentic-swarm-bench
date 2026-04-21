"""ASB Cache Defeat module -- live space-doubling prefix-cache invalidation.

Proprietary module for AgenticSwarmBench. Provides:
  - Scenario-level LCP-aware poisoning (poison_task_execution)
  - Per-message space-doubling for speed benchmarks (poison_messages)
"""

from __future__ import annotations

from asb_cache_defeat.live_poison import (
    compute_lcp,
    compute_scenario_lcp,
    find_isolated_spaces,
    generate_poison_mask,
    poison_task_execution,
)
from asb_cache_defeat.padding_poison import poison_messages, poison_text_spaces

__all__ = [
    "compute_lcp",
    "compute_scenario_lcp",
    "find_isolated_spaces",
    "generate_poison_mask",
    "poison_messages",
    "poison_task_execution",
    "poison_text_spaces",
]


class CacheDefeatModule:
    """Entry-point class for the ASB plugin system."""

    name = "cache_defeat"

    def register_cli_flags(self, group):
        pass

    def apply(self, context: dict):
        pass
