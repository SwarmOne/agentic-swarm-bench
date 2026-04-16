"""Shared test fixtures and helpers."""

from __future__ import annotations

import json

import pytest

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult


@pytest.fixture
def minimal_config():
    return BenchmarkConfig(endpoint="http://test:8000", model="test-model")


@pytest.fixture
def simple_run():
    run = BenchmarkRun(
        model="test-model", endpoint="http://test:8000", started_at="2026-01-01T00:00:00"
    )
    scenario = ScenarioResult(
        num_users=1,
        context_profile="fresh",
        context_tokens=6000,
        wall_time_s=1.0,
        requests=[
            RequestMetrics(request_id=1, completion_tokens=20, tok_per_sec=20.0, ttft_ms=50.0)
        ],
    )
    run.scenarios.append(scenario)
    return run


def make_sse_content(*chunks: dict, done: bool = True) -> bytes:
    """Build raw SSE bytes from a list of OpenAI-style chunk dicts."""
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def make_content_chunk(text: str = "hi", finish: str | None = None) -> dict:
    chunk: dict = {"choices": [{"delta": {"content": text}}]}
    if finish:
        chunk["choices"][0]["finish_reason"] = finish
    return chunk


def make_reasoning_chunk(text: str = "thinking") -> dict:
    return {"choices": [{"delta": {"reasoning_content": text}}]}


@pytest.fixture
def minimal_scenario(tmp_path):
    """A one-task, one-entry scenario backed by a tmp JSONL file."""
    entry = {
        "seq": 1,
        "experiment_id": "test-exp",
        "timestamp": "2026-01-01T00:00:00Z",
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "max_tokens": 100,
        "stream": True,
    }
    jsonl = tmp_path / "recording.jsonl"
    jsonl.write_text(json.dumps(entry) + "\n")
    return str(jsonl)
