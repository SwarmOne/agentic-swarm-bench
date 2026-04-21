"""Shared pytest markers for optional module availability."""

from __future__ import annotations

import pytest

from agentic_swarm_bench.modules import has_module

requires_cache_defeat = pytest.mark.skipif(
    not has_module("cache_defeat"),
    reason="asb-cache-defeat module not installed",
)
requires_scheduler = pytest.mark.skipif(
    not has_module("scheduler"),
    reason="asb-scheduler module not installed",
)

_has_evaluator = False
try:
    from agentic_swarm_bench.scenarios import evaluator  # noqa: F401
    _has_evaluator = True
except ImportError:
    pass

requires_evaluator = pytest.mark.skipif(
    not _has_evaluator,
    reason="evaluator module not available",
)
