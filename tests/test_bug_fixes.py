"""Regression tests for the 4 bugs found during hands-on testing.

Bug 1: list-tasks/list-scenarios --format json emitted invalid JSON
       (Rich console.print wraps lines, inserting bare newlines inside strings).
Bug 2: asb compare showed "Tied" when one side had zero successful requests.
Bug 3: ASB_ENDPOINT/ASB_MODEL env vars and YAML endpoint/model not accepted
       (Click required=True blocked them before build_config could inject them).
Bug 4: asb list-tasks -t p999 silently returned all 110 tasks instead of erroring.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from agentic_swarm_bench.cli import main
from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_swarm_bench.report.markdown import generate_comparison

# ---------------------------------------------------------------------------
# Bug 1 – JSON output is valid JSON
# ---------------------------------------------------------------------------


def test_list_tasks_json_is_valid():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--format", "json"])
    assert result.exit_code == 0, result.output
    tasks = json.loads(result.output)
    assert isinstance(tasks, list)
    assert len(tasks) == 110
    assert tasks[0]["id"] == "P1"


def test_list_scenarios_json_is_valid():
    runner = CliRunner()
    result = runner.invoke(main, ["list-scenarios", "--format", "json"])
    assert result.exit_code == 0, result.output
    scenarios = json.loads(result.output)
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 1
    names = [s["name"] for s in scenarios]
    assert "markdown-note-app" in names


def test_list_tasks_json_no_embedded_newlines():
    """The specific failure mode: Rich wraps long strings with bare newlines."""
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--format", "json"])
    assert result.exit_code == 0
    # If any bare newline appears inside a JSON string value, json.loads raises
    parsed = json.loads(result.output)
    assert len(parsed) > 0


# ---------------------------------------------------------------------------
# Bug 2 – compare winner summary with one-sided failures
# ---------------------------------------------------------------------------


def _make_run(model: str, tok_per_sec: float, profile: str = "fresh", ok: int = 5):
    run = BenchmarkRun(model=model, endpoint="http://test", started_at="2026-01-01T00:00:00")
    reqs = []
    for i in range(ok):
        reqs.append(
            RequestMetrics(
                request_id=i,
                user_id=0,
                task_id=f"P{i+1}",
                context_profile=profile,
                context_tokens=6000,
                ttft_ms=500.0,
                total_time_s=1.0,
                completion_tokens=50,
                tok_per_sec=tok_per_sec,
                prefill_tok_per_sec=5000.0,
            )
        )
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile=profile,
            context_tokens=6000,
            wall_time_s=5.0,
            requests=reqs,
        )
    )
    return run


def _make_failed_run(model: str, profile: str = "fresh"):
    """A run where all requests failed (error field set, no metrics)."""
    run = BenchmarkRun(model=model, endpoint="http://test", started_at="2026-01-01T00:00:00")
    reqs = [
        RequestMetrics(
            request_id=0,
            user_id=0,
            task_id="P1",
            context_profile=profile,
            context_tokens=6000,
            error="HTTP 500: Internal Server Error",
        )
    ]
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile=profile,
            context_tokens=6000,
            wall_time_s=1.0,
            requests=reqs,
        )
    )
    return run


def test_compare_candidate_full_failure_not_tied():
    """When candidate has zero successes, baseline should win - not 'Tied'."""
    baseline = _make_run("fast-model", tok_per_sec=60.0, profile="fresh")
    candidate = _make_failed_run("broken-model", profile="fresh")

    report = generate_comparison(baseline, candidate)

    assert "Tied" not in report
    assert "Baseline" in report  # baseline wins


def test_compare_one_sided_failure_excluded_from_count():
    """Scenario with 0 successes on one side is excluded from the win count."""
    # baseline wins fresh, candidate fails on short
    baseline_fresh = _make_run("model-a", tok_per_sec=50.0, profile="fresh")
    baseline_fresh.scenarios.append(
        _make_run("model-a", tok_per_sec=45.0, profile="short").scenarios[0]
    )
    candidate_fresh = _make_run("model-b", tok_per_sec=30.0, profile="fresh")
    failed_short = _make_failed_run("model-b", profile="short")
    candidate_fresh.scenarios.append(failed_short.scenarios[0])

    report = generate_comparison(baseline_fresh, candidate_fresh)

    # baseline wins fresh (50 > 30), short is excluded (candidate failed)
    # so baseline wins 1/1 valid scenario, short is noted as excluded
    assert "Baseline" in report
    assert "excluded" in report or "zero completions" in report
    assert "Tied" not in report


def test_compare_both_fail_no_valid_scenarios():
    """If both sides fail on all scenarios, report no valid comparisons."""
    run_a = _make_failed_run("model-a")
    run_b = _make_failed_run("model-b")

    report = generate_comparison(run_a, run_b)

    assert "No valid comparison scenarios" in report
    assert "Tied" not in report


def test_compare_honest_tie_still_works():
    """A genuine tie (both succeed, same tok/s) still says Tied."""
    run_a = _make_run("model-a", tok_per_sec=50.0)
    run_b = _make_run("model-b", tok_per_sec=50.0)

    report = generate_comparison(run_a, run_b)

    assert "Tied" in report


# ---------------------------------------------------------------------------
# Bug 3 – ASB_ENDPOINT / ASB_MODEL env vars respected by CLI
# ---------------------------------------------------------------------------


def test_speed_env_vars_accepted(monkeypatch):
    """ASB_ENDPOINT and ASB_MODEL env vars should satisfy the endpoint/model requirement."""
    monkeypatch.setenv("ASB_ENDPOINT", "http://env-server:8000")
    monkeypatch.setenv("ASB_MODEL", "env-model")
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "--dry-run", "-t", "p1-p3", "-u", "1", "-p", "fresh"])
    assert result.exit_code == 0, result.output
    assert "env-model" in result.output
    assert "env-server" in result.output


def test_speed_missing_endpoint_shows_helpful_error():
    """Without endpoint on CLI or in env, show a clear error (not Click's generic one)."""
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "-m", "some-model", "--dry-run", "-t", "p1"])
    assert result.exit_code != 0
    assert "endpoint" in result.output.lower()


def test_speed_missing_model_shows_helpful_error():
    """Without model on CLI or in env, show a clear error."""
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "-e", "http://server:8000", "--dry-run", "-t", "p1"])
    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_yaml_config_endpoint_model_accepted(tmp_path):
    """endpoint and model from a YAML config file should satisfy the requirement."""
    config_file = tmp_path / "bench.yml"
    config_file.write_text("endpoint: http://yaml-server:9000\nmodel: yaml-model\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--config", str(config_file), "speed", "--dry-run", "-t", "p1", "-u", "1", "-p", "fresh"],
    )
    assert result.exit_code == 0, result.output
    assert "yaml-model" in result.output
    assert "yaml-server" in result.output


# ---------------------------------------------------------------------------
# Bug 4 – Invalid task range raises UsageError (not silent fallback)
# ---------------------------------------------------------------------------


def test_list_tasks_invalid_range_errors():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "-t", "p999"])
    assert result.exit_code != 0
    assert "No tasks matched" in result.output


def test_list_tasks_invalid_tags_errors():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--tags", "nonexistent_lang_xyz"])
    assert result.exit_code != 0
    assert "No tasks matched" in result.output


def test_list_tasks_no_filter_still_works():
    """list-tasks with no filter should still return all 110 tasks (no regression)."""
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks"])
    assert result.exit_code == 0
    assert "110 total" in result.output


def test_list_tasks_valid_range_works():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "-t", "p1-p10"])
    assert result.exit_code == 0
    assert "10 total" in result.output
