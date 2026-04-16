"""Tests for the scenario recording proxy in scenarios/recorder.py."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
respx_mod = pytest.importorskip("respx")

import httpx  # noqa: E402
import respx  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agentic_swarm_bench.proxy.utils import _detect_upstream_api  # noqa: E402
from agentic_swarm_bench.scenarios.recorder import create_recording_app  # noqa: E402

# ---------------------------------------------------------------------------
# _detect_upstream_api  (shared implementation)
# ---------------------------------------------------------------------------


def test_detect_explicit_wins():
    assert _detect_upstream_api("https://api.anthropic.com", "openai") == "openai"


def test_detect_anthropic_host():
    assert _detect_upstream_api("https://api.anthropic.com", None) == "anthropic"


def test_detect_openai_fallback():
    assert _detect_upstream_api("http://localhost:11434", None) == "openai"


# ---------------------------------------------------------------------------
# create_recording_app
# ---------------------------------------------------------------------------


def test_create_recording_app_returns_fastapi(tmp_path):
    app = create_recording_app(
        upstream_url="http://fake:8000",
        model="test",
        output_file=str(tmp_path / "recording.jsonl"),
    )
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# /recording/status
# ---------------------------------------------------------------------------


async def test_recording_status_initial(tmp_path):
    app = create_recording_app(
        upstream_url="http://fake:8000",
        model="test",
        output_file=str(tmp_path / "recording.jsonl"),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recording/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "experiment_id" in data
    assert data["requests_recorded"] == 0


# ---------------------------------------------------------------------------
# Non-API path passthrough (does NOT record)
# ---------------------------------------------------------------------------


async def test_non_api_path_is_passthrough(tmp_path):
    output_file = tmp_path / "recording.jsonl"
    app = create_recording_app(
        upstream_url="http://fake-upstream:8000",
        model="test",
        output_file=str(output_file),
    )
    with respx.mock:
        respx.get("http://fake-upstream:8000/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")

    assert resp.status_code == 200
    # Passthrough path should NOT write to the recording file
    assert not output_file.exists() or output_file.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Messages request writes a JSONL entry
# ---------------------------------------------------------------------------


async def test_openai_chat_completions_records_entry(tmp_path):
    output_file = tmp_path / "recording.jsonl"
    app = create_recording_app(
        upstream_url="http://fake-upstream:8000",
        model="test",
        output_file=str(output_file),
    )

    fake_response = {
        "id": "cmpl-1",
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with respx.mock:
        respx.post("http://fake-upstream:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=fake_response)
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            body = {
                "model": "test",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 50,
                "stream": False,
            }
            resp = await client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    assert output_file.exists()
    lines = [ln for ln in output_file.read_text().strip().splitlines() if ln]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["seq"] == 1
    assert len(entry["messages"]) >= 1
