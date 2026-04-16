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
