"""Tests for the speed benchmark orchestration in runner/direct.py."""

from __future__ import annotations

import json

import pytest

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_swarm_bench.runner.direct import _get_cache_passes, _save_outputs, run_speed_benchmark

# ---------------------------------------------------------------------------
# _get_cache_passes
# ---------------------------------------------------------------------------


def test_cold_cache_defeats():
    passes = _get_cache_passes("cold")
    assert passes == [("cold", True)]


def test_warm_cache_no_defeat():
    passes = _get_cache_passes("warm")
    assert passes == [("warm", False)]


def test_both_cache_returns_two_passes():
    passes = _get_cache_passes("both")
    assert len(passes) == 2
    labels = {p[0] for p in passes}
    defeats = {p[1] for p in passes}
    assert "cold" in labels
    assert "warm" in labels
    assert True in defeats
    assert False in defeats


def test_unknown_cache_mode_defaults_cold():
    passes = _get_cache_passes("unknown")
    assert passes == [("cold", True)]


# ---------------------------------------------------------------------------
# run_speed_benchmark: SystemExit(1) on empty resolved_scenarios
# ---------------------------------------------------------------------------


async def test_empty_scenarios_raises_system_exit():
    """model_context_length=1 excludes every profile (all > 1 token)."""
    config = BenchmarkConfig(
        endpoint="http://test:8000",
        model="test-model",
        model_context_length=1,
    )
    with pytest.raises(SystemExit) as exc:
        await run_speed_benchmark(config)
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _save_outputs: file matrix
# ---------------------------------------------------------------------------


def _make_run():
    run = BenchmarkRun(model="test", endpoint="http://test:8000", started_at="2026-01-01T00:00:00")
    scenario = ScenarioResult(
        num_users=1,
        context_profile="fresh",
        context_tokens=6000,
        wall_time_s=1.0,
        requests=[RequestMetrics(request_id=1, completion_tokens=10, tok_per_sec=10.0)],
    )
    run.scenarios.append(scenario)
    return run


def test_save_outputs_json_extension(tmp_path):
    output = str(tmp_path / "results.json")
    config = BenchmarkConfig(output=output, output_format="json")
    _save_outputs(config, _make_run())
    assert (tmp_path / "results.json").exists()
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["model"] == "test"


def test_save_outputs_md_extension_writes_both(tmp_path):
    output = str(tmp_path / "results.md")
    config = BenchmarkConfig(output=output)
    _save_outputs(config, _make_run())
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.json").exists()
    md_content = (tmp_path / "results.md").read_text()
    assert len(md_content) > 0



def test_save_outputs_bare_name_appends_json(tmp_path):
    output = str(tmp_path / "results")
    config = BenchmarkConfig(output=output)
    _save_outputs(config, _make_run())
    assert (tmp_path / "results.json").exists()
