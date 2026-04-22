"""Tests for CLI command registration and basic flag behaviour."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from agentic_swarm_bench.cli import _require_endpoint_model, main

RUNNER = CliRunner()

ALL_COMMANDS = [
    "speed",
    "eval",
    "agent",
    "list-tasks",
    "record",
    "replay",
    "list-scenarios",
    "report",
    "compare",
]


def test_version():
    result = RUNNER.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "agentic-swarm-bench" in result.output.lower() or result.output.strip()


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_help_exits_zero(cmd):
    result = RUNNER.invoke(main, [cmd, "--help"])
    assert result.exit_code == 0, f"{cmd} --help failed:\n{result.output}"
    assert "Usage:" in result.output or "usage:" in result.output.lower()


def test_speed_dry_run_no_requests():
    """--dry-run with valid endpoint + model prints a plan and exits 0, no HTTP."""
    result = RUNNER.invoke(
        main,
        [
            "speed",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output


def test_require_endpoint_model_missing_endpoint():
    import click

    with pytest.raises(click.UsageError, match="endpoint"):
        _require_endpoint_model("", "some-model")


def test_require_endpoint_model_missing_model():
    import click

    with pytest.raises(click.UsageError, match="model"):
        _require_endpoint_model("http://host", "")


def test_require_endpoint_model_both_set():
    _require_endpoint_model("http://host", "my-model")  # must not raise


def test_list_tasks_no_network():
    result = RUNNER.invoke(main, ["list-tasks"])
    assert result.exit_code == 0


def test_list_scenarios_no_network():
    result = RUNNER.invoke(main, ["list-scenarios"])
    assert result.exit_code == 0


@pytest.mark.parametrize("mode", ["allcold", "allwarm", "realistic"])
def test_speed_cache_mode_accepted(mode):
    result = RUNNER.invoke(
        main,
        [
            "speed",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--dry-run",
            "--cache-mode", mode,
        ],
    )
    assert result.exit_code == 0, f"cache-mode={mode} failed:\n{result.output}"


def test_speed_old_cache_mode_cold_rejected():
    """Old 'cold' name is no longer valid; Click should return an error."""
    result = RUNNER.invoke(
        main,
        [
            "speed",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--dry-run",
            "--cache-mode", "cold",
        ],
    )
    assert result.exit_code != 0


@pytest.mark.parametrize("mode", ["realistic", "allcold", "allwarm"])
def test_replay_cache_mode_in_help(mode):
    """All three cache modes must appear in replay --help."""
    result = RUNNER.invoke(main, ["replay", "--help"])
    assert result.exit_code == 0
    assert mode in result.output


def test_replay_users_flag_rejected_with_migration_hint():
    """`--users N` on replay must hard-fail with a message pointing to --repetitions.

    The flag was removed because all N "users" sent byte-identical poisoned
    payloads, so users 1..N-1 rode the KV cache for free and cache hit-rate
    was artificially inflated. We keep it hidden just to produce a precise
    error instead of Click's generic "no such option".
    """
    result = RUNNER.invoke(
        main,
        [
            "replay",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--scenario", "noop",
            "--users", "8",
        ],
    )
    assert result.exit_code != 0
    assert "--users" in result.output
    assert "--repetitions" in result.output
    assert "--max-concurrent" in result.output


def test_replay_users_flag_hidden_from_help():
    """The deprecated --users flag must not appear in `asb replay --help`."""
    result = RUNNER.invoke(main, ["replay", "--help"])
    assert result.exit_code == 0
    assert "--users" not in result.output


def test_replay_repetitions_max_concurrent_dry_run(tmp_path):
    """--repetitions + --max-concurrent (the new recipe for 'N concurrent users') works.

    Uses an inline scenario file so the test isn't coupled to built-in
    scenarios that may not be shipped to CI.
    """
    import json

    scenario = tmp_path / "scenario.jsonl"
    scenario.write_text(
        json.dumps({
            "seq": 1,
            "experiment_id": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "max_tokens": 100,
            "stream": True,
        })
        + "\n"
    )

    result = RUNNER.invoke(
        main,
        [
            "replay",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--scenario", str(scenario),
            "--repetitions", "3",
            "--max-concurrent", "3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"dry-run failed:\n{result.output}"
    assert "DRY RUN" in result.output
    assert "Users" not in result.output


def test_replay_seed_flag_in_help():
    """--seed must be documented in replay --help for reproducible random order."""
    result = RUNNER.invoke(main, ["replay", "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.output


def test_agent_schedule_flags_in_help():
    """agent command must expose --repetitions, --max-concurrent, --policy, --seed."""
    result = RUNNER.invoke(main, ["agent", "--help"])
    assert result.exit_code == 0
    for flag in ("--repetitions", "--max-concurrent", "--policy", "--seed"):
        assert flag in result.output, f"missing {flag} in agent --help"


# ---------------------------------------------------------------------------
# Task range fallback warnings and dry-run label accuracy
# ---------------------------------------------------------------------------


def test_speed_empty_task_range_warns():
    """When --tasks matches nothing, a warning should be printed before fallback."""
    result = RUNNER.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "--tasks", "p999",
         "--dry-run", "-p", "fresh", "-u", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "P1-P25" in result.output or "p1-p25" in result.output


def test_speed_suite_with_profile_warns():
    result = RUNNER.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "--suite", "quick",
         "--context-profile", "medium", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "medium" in result.output


def test_speed_dry_run_sample_uses_actual_task_id():
    """Sample request label should use the actual first task, not hardcoded P1."""
    result = RUNNER.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "--tasks", "p50-p60",
         "--dry-run", "-p", "fresh", "-u", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "P50 at" in result.output
    assert "P1 at 6K" not in result.output


def test_speed_dry_run_sample_uses_actual_context():
    """Sample request label should use the actual scenario context size."""
    result = RUNNER.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "--tasks", "p1-p3",
         "--dry-run", "-p", "long", "-u", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "P1 at 70K" in result.output


