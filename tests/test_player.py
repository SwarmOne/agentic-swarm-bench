"""Tests for player.py: token estimation, SSE streaming, retry logic, and reasoning fields."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_swarm_bench.metrics.collector import RequestMetrics
from agentic_swarm_bench.scenarios.player import (
    _bucket_label,
    _compute_bucket_wall_time,
    _estimate_tokens,
    _replay_one_request,
    _slice_entries,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Token estimation from content length (~4 chars per token)."""

    def test_empty_string_returns_zero(self):
        assert _estimate_tokens("") == 0

    def test_none_returns_zero(self):
        assert _estimate_tokens(None) == 0

    def test_short_string_returns_minimum_one(self):
        assert _estimate_tokens("hi") == 1
        assert _estimate_tokens("a") == 1
        assert _estimate_tokens("abc") == 1

    def test_exactly_four_chars_returns_one(self):
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_returns_two(self):
        assert _estimate_tokens("abcdefgh") == 2

    def test_long_content_scales_linearly(self):
        text = "x" * 400
        assert _estimate_tokens(text) == 100

    def test_realistic_english_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = _estimate_tokens(text)
        assert tokens == len(text) // 4
        assert tokens >= 10

    def test_single_sse_chunk_with_many_tokens(self):
        """Anthropic sends 20+ tokens per SSE chunk. This must count > 1."""
        chunk_content = "Here is a response with many tokens in a single SSE event"
        tokens = _estimate_tokens(chunk_content)
        assert tokens > 10

    def test_reasoning_content_also_estimated(self):
        reasoning = "Let me think through this step by step. First, I need to consider..."
        tokens = _estimate_tokens(reasoning)
        assert tokens > 10


# ---------------------------------------------------------------------------
# _replay_one_request -- SSE streaming with token counting
# ---------------------------------------------------------------------------


def _make_sse_lines(chunks: list[dict], usage: dict | None = None) -> list[str]:
    """Build SSE `data: ...` lines from chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    if usage:
        lines.append(f"data: {json.dumps({'usage': usage, 'choices': []})}")
    lines.append("data: [DONE]")
    return lines


def _sse_chunk(content: str | None = None, reasoning: str | None = None) -> dict:
    """Build a single SSE chunk with optional content/reasoning."""
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta}]}


def _sse_chunk_reasoning_field(reasoning: str) -> dict:
    """Build an SSE chunk using the 'reasoning' field (Together/GLM convention)."""
    return {"choices": [{"delta": {"reasoning": reasoning}}]}


class FakeStreamResponse:
    """Fake httpx streaming response for testing _replay_one_request."""

    def __init__(self, lines: list[str], status_code: int = 200, body: bytes = b""):
        self.lines = lines
        self.status_code = status_code
        self._body = body

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    """Fake httpx.AsyncClient that returns a FakeStreamResponse from .stream()."""

    def __init__(self, response: FakeStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


def _replay(**kwargs):
    """Shorthand: call _replay_one_request with sensible defaults, run sync."""
    defaults = dict(
        client=None,
        url="http://test/v1/chat/completions",
        model="test",
        headers={},
        messages=[{"role": "user", "content": "test"}],
        max_tokens=512,
        seq=0,
        timeout=30.0,
    )
    defaults.update(kwargs)
    return _run(_replay_one_request(**defaults))


def test_token_count_uses_content_length_not_chunk_count():
    """The old bug: each SSE chunk counted as 1 token.
    With 3 chunks of ~60 chars each, we should get ~45 tokens, not 3."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="Here is a detailed response with many tokens "),
            _sse_chunk(content="continuing the explanation with more details "),
            _sse_chunk(content="and finishing up with a final conclusion here"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens > 3, (
        f"Got {m.completion_tokens} tokens - still counting chunks as 1 each"
    )
    assert m.completion_tokens > 30


def test_usage_completion_tokens_overrides_estimate():
    """When the API provides usage.completion_tokens, it should override the estimate."""
    lines = _make_sse_lines(
        [_sse_chunk(content="short response")],
        usage={"prompt_tokens": 10, "completion_tokens": 71},
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 71
    assert m.prompt_tokens == 10


def test_no_usage_falls_back_to_estimate():
    """Without usage data, fall back to the char-based estimate."""
    text = "a" * 80  # 80 chars -> 20 estimated tokens
    lines = _make_sse_lines([_sse_chunk(content=text)])
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 20


def test_stream_options_include_usage_in_payload():
    """Payload should include stream_options: {include_usage: true}."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(client=CapturingClient())
    assert "stream_options" in captured_payload
    assert captured_payload["stream_options"]["include_usage"] is True


# ---------------------------------------------------------------------------
# Reasoning token field name handling
# ---------------------------------------------------------------------------


def test_reasoning_content_field():
    """Anthropic/DeepSeek convention: delta.reasoning_content."""
    lines = _make_sse_lines(
        [
            _sse_chunk(reasoning="Let me think about this problem carefully"),
            _sse_chunk(content="The answer is 42"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens > 0
    assert m.ttft_thinking_ms > 0
    assert m.ttft_visible_ms > 0


def test_reasoning_field_together_convention():
    """Together/GLM convention: delta.reasoning (not reasoning_content)."""
    lines = _make_sse_lines(
        [
            _sse_chunk_reasoning_field("Step 1: analyze the problem carefully"),
            _sse_chunk(content="Final answer is here"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens > 0, "Together-style 'reasoning' field was not counted"
    assert m.ttft_thinking_ms > 0


def test_no_reasoning_tokens_when_content_only():
    """Regular (non-reasoning) response should have zero thinking tokens."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="Just a normal response without thinking"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens == 0
    assert m.ttft_thinking_ms == 0


# ---------------------------------------------------------------------------
# Rate limit retry (HTTP 429)
# ---------------------------------------------------------------------------


def test_http_429_records_error_when_retries_disabled():
    """With max_retries=0 (default), 429 is recorded as an error, no retry."""
    resp = FakeStreamResponse([], status_code=429, body=b"rate limited")
    m = _replay(client=FakeClient(resp), max_retries=0)
    assert m.error is not None
    assert "429" in m.error


def test_http_429_retries_then_succeeds():
    """With retries enabled, 429 should be retried with backoff."""
    call_count = 0

    class RetryClient:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeStreamResponse([], status_code=429, body=b"rate limited")
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="success after retry")]))

    with patch("agentic_swarm_bench.scenarios.player.asyncio.sleep", new_callable=AsyncMock):
        m = _replay(client=RetryClient(), max_retries=2)
    assert call_count == 2
    assert m.error is None
    assert m.completion_tokens > 0


def test_stream_options_retry_on_rejection():
    """If a provider rejects stream_options, retry without it."""
    call_count = 0
    captured_payloads = []

    class RejectStreamOptionsClient:
        def stream(self, method, url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(dict(json or {}))
            if call_count == 1 and "stream_options" in (json or {}):
                return FakeStreamResponse(
                    [],
                    status_code=400,
                    body=b'{"error": "stream_options is not supported"}',
                )
            return FakeStreamResponse(
                _make_sse_lines([_sse_chunk(content="ok without stream_options")])
            )

    m = _replay(client=RejectStreamOptionsClient(), max_retries=1)
    assert call_count == 2
    assert "stream_options" in captured_payloads[0]
    assert "stream_options" not in captured_payloads[1]
    assert m.error is None


def test_http_500_not_retried():
    """Non-429/non-stream_options errors should not be retried."""
    call_count = 0

    class Error500Client:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeStreamResponse([], status_code=500, body=b"internal error")

    m = _replay(client=Error500Client(), max_retries=3)
    assert call_count == 1
    assert "500" in m.error


def test_429_exhausts_all_retries():
    """If every attempt gets 429, we should see max_retries+1 attempts and an error."""
    call_count = 0

    class Always429Client:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeStreamResponse([], status_code=429, body=b"rate limited")

    with patch("agentic_swarm_bench.scenarios.player.asyncio.sleep", new_callable=AsyncMock):
        m = _replay(client=Always429Client(), max_retries=2)
    assert call_count == 3  # initial + 2 retries
    assert m.error is not None
    assert "429" in m.error


# ---------------------------------------------------------------------------
# Payload construction edge cases
# ---------------------------------------------------------------------------


def test_openai_endpoint_uses_max_completion_tokens():
    """OpenAI endpoints should use max_completion_tokens, not max_tokens."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(
        client=CapturingClient(),
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-5.4",
        max_tokens=1024,
    )
    assert "max_completion_tokens" in captured_payload
    assert "max_tokens" not in captured_payload


def test_non_openai_endpoint_uses_max_tokens():
    """Non-OpenAI endpoints should use max_tokens."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(
        client=CapturingClient(),
        url="https://api.anthropic.com/v1/chat/completions",
        model="claude",
        max_tokens=1024,
    )
    assert "max_tokens" in captured_payload
    assert "max_completion_tokens" not in captured_payload


def test_max_tokens_capped_at_4096():
    """max_tokens should be capped at 4096 regardless of input."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(client=CapturingClient(), max_tokens=16384)
    assert captured_payload["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# _bucket_label
# ---------------------------------------------------------------------------


class TestBucketLabel:
    def test_fresh(self):
        assert _bucket_label(0) == "fresh"
        assert _bucket_label(5000) == "fresh"
        assert _bucket_label(9999) == "fresh"

    def test_short(self):
        assert _bucket_label(10000) == "short"
        assert _bucket_label(29999) == "short"

    def test_medium(self):
        assert _bucket_label(30000) == "medium"
        assert _bucket_label(54999) == "medium"

    def test_long(self):
        assert _bucket_label(55000) == "long"
        assert _bucket_label(84999) == "long"

    def test_full(self):
        assert _bucket_label(85000) == "full"
        assert _bucket_label(149999) == "full"

    def test_xl(self):
        assert _bucket_label(150000) == "xl"
        assert _bucket_label(299999) == "xl"

    def test_xxl(self):
        assert _bucket_label(300000) == "xxl"
        assert _bucket_label(1000000) == "xxl"


# ---------------------------------------------------------------------------
# _compute_bucket_wall_time
# ---------------------------------------------------------------------------


class TestComputeBucketWallTime:
    def test_empty(self):
        assert _compute_bucket_wall_time([], num_users=1) == 0.0

    def test_single_user(self):
        reqs = [
            RequestMetrics(user_id=0, total_time_s=1.0),
            RequestMetrics(user_id=0, total_time_s=2.0),
        ]
        assert _compute_bucket_wall_time(reqs, num_users=1) == 3.0

    def test_multi_user_takes_max(self):
        reqs = [
            RequestMetrics(user_id=0, total_time_s=1.0),
            RequestMetrics(user_id=0, total_time_s=2.0),  # user 0 total: 3.0
            RequestMetrics(user_id=1, total_time_s=1.5),
            RequestMetrics(user_id=1, total_time_s=1.0),  # user 1 total: 2.5
        ]
        assert _compute_bucket_wall_time(reqs, num_users=2) == 3.0


# ---------------------------------------------------------------------------
# _slice_entries
# ---------------------------------------------------------------------------


class TestSliceEntries:
    def _make_entry(self, prompt_tokens=None, content_len=400):
        entry = MagicMock()
        entry.prompt_tokens = prompt_tokens
        entry.messages = [{"content": "x" * content_len}]
        return entry

    def test_no_budget_returns_all(self):
        entries = [self._make_entry() for _ in range(5)]
        result = _slice_entries(entries, slice_tokens=None)
        assert len(result) == 5

    def test_budget_limits_entries(self):
        entries = [self._make_entry(prompt_tokens=1000) for _ in range(10)]
        result = _slice_entries(entries, slice_tokens=2500)
        assert len(result) == 2

    def test_always_includes_first_entry_even_if_exceeds(self):
        entries = [self._make_entry(prompt_tokens=5000)]
        result = _slice_entries(entries, slice_tokens=100)
        assert len(result) == 1

    def test_falls_back_to_char_estimate(self):
        entries = [self._make_entry(prompt_tokens=None, content_len=400) for _ in range(5)]
        result = _slice_entries(entries, slice_tokens=250)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------


def test_ttft_and_tok_per_sec_computed():
    """Verify that TTFT and tok/s are computed on a successful stream."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="hello world response with enough tokens to measure"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.ttft_ms > 0
    assert m.total_time_s > 0
    assert m.error is None


def test_context_tokens_estimated_from_messages():
    """context_tokens should be estimated from message content length."""
    content = "x" * 800  # 200 estimated tokens
    lines = _make_sse_lines([_sse_chunk(content="ok")])
    m = _replay(
        client=FakeClient(FakeStreamResponse(lines)),
        messages=[{"role": "user", "content": content}],
    )
    assert m.context_tokens == 200


def test_timeout_records_error():
    """Wall-clock timeout should produce an error metric."""

    class HangingResponse:
        status_code = 200

        async def aiter_lines(self):
            await asyncio.sleep(999)
            yield "data: [DONE]"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class HangingClient:
        def stream(self, method, url, **kwargs):
            return HangingResponse()

    m = _replay(client=HangingClient(), timeout=0.01)
    assert m.error is not None
    assert "timeout" in m.error.lower() or "Timeout" in m.error


def test_empty_stream_returns_zero_tokens():
    """A stream that sends no content chunks should have 0 completion tokens."""
    lines = _make_sse_lines([])
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 0


def test_mixed_content_and_reasoning_tokens():
    """Both content tokens and reasoning tokens should be counted separately."""
    reasoning_text = "Let me think step by step about this problem"
    content_text = "The answer is forty two"
    lines = _make_sse_lines(
        [
            _sse_chunk(reasoning=reasoning_text),
            _sse_chunk(content=content_text),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    expected_reasoning = _estimate_tokens(reasoning_text)
    expected_total = expected_reasoning + _estimate_tokens(content_text)

    assert m.thinking_tokens == expected_reasoning
    assert m.completion_tokens == expected_total


def test_malformed_json_in_sse_is_skipped():
    """Malformed JSON in SSE should be silently skipped, not crash."""
    lines = [
        "data: {invalid json}",
        f"data: {json.dumps(_sse_chunk(content='valid chunk'))}",
        "data: [DONE]",
    ]
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.error is None
    assert m.completion_tokens > 0
