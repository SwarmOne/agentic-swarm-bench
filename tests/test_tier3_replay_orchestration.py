"""Tests for replay orchestration in scenarios/player.py."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_swarm_bench.scenarios.player import _replay_task_entries, _save_replay_output
from agentic_swarm_bench.scenarios.registry import RecordingEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(content: str = "hello") -> RecordingEntry:
    return RecordingEntry(
        seq=1,
        messages=[{"role": "user", "content": content}],
        model="test",
        max_tokens=100,
    )


def _make_run() -> BenchmarkRun:
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


def _make_scenario_jsonl(tmp_path: Path, messages_content: str = "hello") -> str:
    entry = {
        "seq": 1,
        "experiment_id": "test",
        "timestamp": "2026-01-01T00:00:00Z",
        "messages": [{"role": "user", "content": messages_content}],
        "model": "test-model",
        "max_tokens": 100,
        "stream": True,
    }
    path = tmp_path / "scenario.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# dry_run: returns early without HTTP
# ---------------------------------------------------------------------------


async def test_dry_run_returns_without_http(tmp_path):
    scenario_path = _make_scenario_jsonl(tmp_path)
    config = BenchmarkConfig(
        endpoint="http://fake:8000",
        model="test-model",
        dry_run=True,
    )
    run = await __import__(
        "agentic_swarm_bench.scenarios.player", fromlist=["replay_scenario"]
    ).replay_scenario(config, scenario_path)
    assert isinstance(run, BenchmarkRun)
    assert len(run.scenarios) == 0


# ---------------------------------------------------------------------------
# model_context_length: entries exceeding limit are skipped
# ---------------------------------------------------------------------------


async def test_model_context_length_skips_long_entries():
    # Entry with ~50 chars / 4 = ~12 tokens
    short_entry = _make_entry("hi")  # very short
    # Entry with 2000 chars = 500 tokens
    long_entry = _make_entry("x" * 2000)

    import httpx
    import respx

    with respx.mock:
        # Short entry should get a request; long entry should be skipped
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode()
                + b"\ndata: [DONE]\n",
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient() as client:
            results = await _replay_task_entries(
                client=client,
                url="http://fake:8000/v1/chat/completions",
                model_override="test",
                headers={},
                entries=[short_entry, long_entry],
                timeout=5.0,
                user_id=0,
                model_context_length=50,  # only ~50 tokens allowed
            )

    # long_entry (500 tokens) should be skipped, short_entry (~1 token) should pass
    assert len(results) == 1


# ---------------------------------------------------------------------------
# _save_replay_output
# ---------------------------------------------------------------------------


def test_save_replay_output_json(tmp_path):
    output = str(tmp_path / "results.json")
    config = BenchmarkConfig(output=output)
    _save_replay_output(config, _make_run())
    assert (tmp_path / "results.json").exists()
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["model"] == "test"


def test_save_replay_output_md_writes_both(tmp_path):
    output = str(tmp_path / "results.md")
    config = BenchmarkConfig(output=output)
    _save_replay_output(config, _make_run())
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.json").exists()
    assert len((tmp_path / "results.md").read_text()) > 0


def test_save_replay_output_bare_name_appends_json(tmp_path):
    output = str(tmp_path / "results")
    config = BenchmarkConfig(output=output)
    _save_replay_output(config, _make_run())
    assert (tmp_path / "results.json").exists()
