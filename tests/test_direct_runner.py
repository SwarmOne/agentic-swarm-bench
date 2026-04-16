"""Tests for the SSE streaming hot path in runner/direct.py."""

from __future__ import annotations

import json

import httpx
import respx

from agentic_swarm_bench.runner.direct import _send_streaming_request
from tests.conftest import make_content_chunk, make_reasoning_chunk, make_sse_content

URL = "http://fake-endpoint/v1/chat/completions"


async def _invoke(sse_content: bytes, status: int = 200, **kwargs):
    with respx.mock:
        respx.post(URL).mock(
            return_value=httpx.Response(
                status,
                content=sse_content,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient() as client:
            return await _send_streaming_request(
                client=client,
                url=URL,
                model="test-model",
                headers={},
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                user_id=0,
                task_id="P1",
                context_profile="fresh",
                context_tokens=6000,
                timeout=10.0,
                **kwargs,
            )


async def test_happy_path_five_tokens():
    chunks = [make_content_chunk(f"tok{i}") for i in range(5)]
    sse = make_sse_content(*chunks)
    m = await _invoke(sse)
    assert m.completion_tokens == 5
    assert m.ttft_ms > 0
    assert m.tok_per_sec > 0
    assert m.error is None


async def test_non_200_sets_error():
    m = await _invoke(b"bad request", status=400)
    assert m.error is not None
    assert "400" in m.error
    assert m.completion_tokens == 0


async def test_done_sentinel_terminates_cleanly():
    sse = make_sse_content(make_content_chunk("a"), make_content_chunk("b"))
    m = await _invoke(sse)
    assert m.completion_tokens == 2
    assert m.error is None


async def test_reasoning_tokens_tracked_separately():
    chunks = [
        make_reasoning_chunk("thought 1"),
        make_reasoning_chunk("thought 2"),
        make_content_chunk("visible answer"),
    ]
    sse = make_sse_content(*chunks)
    m = await _invoke(sse)
    assert m.thinking_tokens == 2
    assert m.completion_tokens == 3
    assert m.ttft_thinking_ms > 0
    assert m.ttft_visible_ms > 0
    assert m.ttft_visible_ms >= m.ttft_thinking_ms


async def test_malformed_json_line_skipped():
    lines = (
        b"data: not-valid-json\n\n"
        + b"data: " + json.dumps(make_content_chunk("ok")).encode() + b"\n\n"
        + b"data: [DONE]\n\n"
    )
    m = await _invoke(lines)
    assert m.completion_tokens == 1
    assert m.error is None


async def test_network_exception_sets_error():
    with respx.mock:
        respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        async with httpx.AsyncClient() as client:
            m = await _send_streaming_request(
                client=client,
                url=URL,
                model="test",
                headers={},
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                user_id=0,
                task_id="T1",
                context_profile="fresh",
                context_tokens=6000,
                timeout=10.0,
            )
    assert m.error is not None
    assert "ConnectError" in m.error


async def test_on_complete_callback_called():
    called = []
    sse = make_sse_content(make_content_chunk("hi"))
    with respx.mock:
        respx.post(URL).mock(
            return_value=httpx.Response(
                200, content=sse, headers={"content-type": "text/event-stream"}
            )
        )
        async with httpx.AsyncClient() as client:
            await _send_streaming_request(
                client=client,
                url=URL,
                model="test",
                headers={},
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                user_id=0,
                task_id="T1",
                context_profile="fresh",
                context_tokens=6000,
                timeout=10.0,
                on_complete=lambda: called.append(1),
            )
    assert len(called) == 1


async def test_usage_chunk_sets_prompt_tokens():
    usage_chunk = {"choices": [], "usage": {"prompt_tokens": 999}}
    sse = make_sse_content(make_content_chunk("hi"), usage_chunk)
    m = await _invoke(sse)
    assert m.prompt_tokens == 999
