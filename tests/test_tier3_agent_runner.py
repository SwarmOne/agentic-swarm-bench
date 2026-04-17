"""Tests for the Claude Code agent orchestration in runner/claude_code.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import respx

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.runner.claude_code import (
    _cleanup_workdir,
    _preflight_check,
    _start_proxy,
    _stop_proxy,
)

# ---------------------------------------------------------------------------
# _preflight_check
# ---------------------------------------------------------------------------


async def test_preflight_404_returns_false():
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(404)
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is False


async def test_preflight_200_returns_true():
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is True


async def test_preflight_400_returns_true():
    """Any non-404 response means the path exists; preflight passes."""
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": "bad model"})
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is True


async def test_preflight_connect_error_returns_false():
    with respx.mock:
        respx.post("http://unreachable:9999/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await _preflight_check("http://unreachable:9999", "openai")
    assert result is False


async def test_preflight_anthropic_path():
    with respx.mock:
        respx.post("http://fake:8000/v1/messages").mock(
            return_value=httpx.Response(200, json={})
        )
        result = await _preflight_check("http://fake:8000", "anthropic")
    assert result is True


async def test_preflight_404_anthropic_returns_false():
    with respx.mock:
        respx.post("http://fake:8000/v1/messages").mock(
            return_value=httpx.Response(404)
        )
        result = await _preflight_check("http://fake:8000", "anthropic")
    assert result is False


# ---------------------------------------------------------------------------
# _cleanup_workdir
# ---------------------------------------------------------------------------


def test_cleanup_removes_logs_when_keep_false(tmp_path):
    log = tmp_path / "slot0_p1_r0.log"
    log.write_text("some log")
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert not log.exists()
    assert not task_dir.exists()


def test_cleanup_keeps_logs_when_keep_true(tmp_path):
    log = tmp_path / "slot0_p1_r0.log"
    log.write_text("some log")
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()

    _cleanup_workdir(tmp_path, keep_logs=True)

    assert log.exists()
    assert task_dir.exists()


def test_cleanup_keeps_non_empty_task_dirs(tmp_path):
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()
    (task_dir / "output.txt").write_text("data")

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert task_dir.exists()


def test_cleanup_keeps_metrics_and_summary(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}")
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "slot0_p1_r0.log").write_text("log")

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "slot0_p1_r0.log").exists()


# ---------------------------------------------------------------------------
# _stop_proxy (now async; wraps asyncio.subprocess.Process)
# ---------------------------------------------------------------------------


async def test_stop_proxy_terminates_proc():
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)
    await _stop_proxy(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_awaited_once()


async def test_stop_proxy_skips_when_already_exited():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    await _stop_proxy(mock_proc)
    mock_proc.terminate.assert_not_called()


async def test_stop_proxy_kills_on_timeout():
    """If proxy doesn't exit within 5s, _stop_proxy must SIGKILL it."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    async def _fake_wait_for(coro, timeout):
        # Close the coroutine we were handed so it never produces an
        # "unawaited coroutine" warning, then simulate the timeout.
        coro.close()
        raise asyncio.TimeoutError()

    with patch(
        "agentic_swarm_bench.runner.claude_code.asyncio.wait_for",
        new=_fake_wait_for,
    ):
        await _stop_proxy(mock_proc)

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _start_proxy (now async)
# ---------------------------------------------------------------------------


async def test_start_proxy_returns_none_when_fastapi_missing():
    config = BenchmarkConfig(endpoint="http://fake:8000", model="test", proxy_port=19001)
    # Setting a sys.modules key to None causes `import` of that name to raise ImportError.
    with patch.dict("sys.modules", {"agentic_swarm_bench.proxy.server": None}):
        result = await _start_proxy(config)

    assert result is None
