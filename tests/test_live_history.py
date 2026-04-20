"""Tests for live-history replay: delta extraction, response capture, and the full loop."""

from __future__ import annotations

import asyncio
import json

from agentic_swarm_bench.scenarios.player import (
    _extract_new_client_messages,
    _get_recorded_assistant_message,
    _replay_one_request,
    _replay_task_entries,
    _replay_task_entries_live,
)
from agentic_swarm_bench.scenarios.registry import RecordingEntry


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers: fake streaming server
# ---------------------------------------------------------------------------


def _sse_chunk(content: str) -> dict:
    return {"choices": [{"delta": {"content": content}}]}


def _make_sse_lines(chunks: list[dict], usage: dict | None = None) -> list[str]:
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    if usage:
        lines.append(f"data: {json.dumps({'usage': usage, 'choices': []})}")
    lines.append("data: [DONE]")
    return lines


class FakeStreamResponse:
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
    def __init__(self, response: FakeStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


class SequentialFakeClient:
    """Returns a different FakeStreamResponse for each successive .stream() call."""

    def __init__(self, responses: list[FakeStreamResponse]):
        self._responses = list(responses)
        self._call_index = 0
        self.captured_messages: list[list[dict]] = []

    def stream(self, method, url, **kwargs):
        payload = kwargs.get("json", {})
        self.captured_messages.append(payload.get("messages", []))
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return resp


# ---------------------------------------------------------------------------
# Helpers: make entries for a growing conversation
# ---------------------------------------------------------------------------


def _make_growing_entries(n: int = 3) -> list[RecordingEntry]:
    """Build entries that mirror a real multi-turn recording.

    Entry 1: [system, user1]
    Entry 2: [system, user1, assistant1, user2]
    Entry 3: [system, user1, assistant1, user2, assistant2, user3]
    """
    conversation: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    entries = []
    for i in range(n):
        conversation = conversation + [{"role": "user", "content": f"Question {i + 1}"}]
        entries.append(
            RecordingEntry(
                seq=i + 1,
                experiment_id="test",
                messages=[dict(m) for m in conversation],
                max_tokens=100,
            )
        )
        conversation = conversation + [
            {"role": "assistant", "content": f"Recorded answer {i + 1}"}
        ]
    return entries


# =========================================================================
# _extract_new_client_messages
# =========================================================================


class TestExtractNewClientMessages:
    def test_first_entry_returns_all_messages(self):
        entry = RecordingEntry(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        )
        result = _extract_new_client_messages(entry, None)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_standard_delta_extracts_user_only(self):
        entries = _make_growing_entries(2)
        result = _extract_new_client_messages(entries[1], entries[0])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Question 2"

    def test_tool_use_delta_keeps_tool_results(self):
        prev = RecordingEntry(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "use a tool"},
            ]
        )
        current = RecordingEntry(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "use a tool"},
                {"role": "assistant", "content": "calling tool..."},
                {"role": "tool", "content": "tool result here"},
                {"role": "user", "content": "thanks"},
            ]
        )
        result = _extract_new_client_messages(current, prev)
        assert len(result) == 2
        roles = [m["role"] for m in result]
        assert roles == ["tool", "user"]

    def test_identical_entries_returns_empty(self):
        entry = RecordingEntry(
            messages=[{"role": "user", "content": "hello"}]
        )
        result = _extract_new_client_messages(entry, entry)
        assert result == []

    def test_returns_copy_not_reference(self):
        entry = RecordingEntry(
            messages=[{"role": "user", "content": "hello"}]
        )
        result = _extract_new_client_messages(entry, None)
        result[0]["content"] = "mutated"
        assert entry.messages[0]["content"] == "hello"


# =========================================================================
# _get_recorded_assistant_message
# =========================================================================


class TestGetRecordedAssistantMessage:
    def test_first_entry_returns_none(self):
        entry = RecordingEntry(messages=[{"role": "user", "content": "hi"}])
        assert _get_recorded_assistant_message(entry, None) is None

    def test_extracts_assistant_from_delta(self):
        entries = _make_growing_entries(2)
        result = _get_recorded_assistant_message(entries[1], entries[0])
        assert result is not None
        assert result["role"] == "assistant"
        assert result["content"] == "Recorded answer 1"

    def test_no_assistant_in_delta_returns_none(self):
        prev = RecordingEntry(messages=[{"role": "user", "content": "hi"}])
        current = RecordingEntry(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "follow up"},
            ]
        )
        assert _get_recorded_assistant_message(current, prev) is None


# =========================================================================
# response_chunks accumulation -- OpenAI path
# =========================================================================


class TestResponseChunksOpenAI:
    def test_accumulates_content_text(self):
        lines = _make_sse_lines([
            _sse_chunk("Hello "),
            _sse_chunk("world!"),
        ])
        chunks: list[str] = []
        m = _run(_replay_one_request(
            client=FakeClient(FakeStreamResponse(lines)),
            url="http://test/v1/chat/completions",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            response_chunks=chunks,
        ))
        assert "".join(chunks) == "Hello world!"
        assert m.error is None

    def test_none_response_chunks_still_works(self):
        lines = _make_sse_lines([_sse_chunk("hello")])
        m = _run(_replay_one_request(
            client=FakeClient(FakeStreamResponse(lines)),
            url="http://test/v1/chat/completions",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            response_chunks=None,
        ))
        assert m.error is None
        assert m.completion_tokens > 0

    def test_reasoning_not_captured_in_response_chunks(self):
        reasoning_chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        lines = _make_sse_lines([reasoning_chunk, _sse_chunk("visible answer")])
        chunks: list[str] = []
        _run(_replay_one_request(
            client=FakeClient(FakeStreamResponse(lines)),
            url="http://test/v1/chat/completions",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            response_chunks=chunks,
        ))
        text = "".join(chunks)
        assert "thinking" not in text
        assert "visible answer" in text

    def test_error_response_leaves_chunks_empty(self):
        resp = FakeStreamResponse([], status_code=500, body=b"Internal Server Error")
        chunks: list[str] = []
        m = _run(_replay_one_request(
            client=FakeClient(resp),
            url="http://test/v1/chat/completions",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            response_chunks=chunks,
        ))
        assert chunks == []
        assert m.error is not None


# =========================================================================
# response_chunks accumulation -- Anthropic path
# =========================================================================


def _anthropic_sse_bytes(*events: dict) -> bytes:
    """Build raw Anthropic SSE bytes from event dicts."""
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}\n")
    return "\n".join(lines).encode()


class FakeAnthropicStreamResponse:
    def __init__(self, raw_bytes: bytes, status_code: int = 200):
        self._raw_bytes = raw_bytes
        self.status_code = status_code

    async def aiter_bytes(self):
        yield self._raw_bytes

    async def aread(self):
        return self._raw_bytes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeAnthropicClient:
    def __init__(self, response: FakeAnthropicStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


class TestResponseChunksAnthropic:
    def test_accumulates_text_delta(self):
        events = _anthropic_sse_bytes(
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world!"}},
            {"type": "message_delta", "usage": {"output_tokens": 2}},
        )
        chunks: list[str] = []
        m = _run(_replay_one_request(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            url="http://test",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            upstream_api="anthropic",
            response_chunks=chunks,
        ))
        assert "".join(chunks) == "Hello world!"
        assert m.error is None

    def test_thinking_delta_not_captured(self):
        events = _anthropic_sse_bytes(
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "answer"}},
            {"type": "message_delta", "usage": {"output_tokens": 2}},
        )
        chunks: list[str] = []
        _run(_replay_one_request(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            url="http://test",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            upstream_api="anthropic",
            response_chunks=chunks,
        ))
        assert "".join(chunks) == "answer"

    def test_none_response_chunks_backward_compat(self):
        events = _anthropic_sse_bytes(
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "usage": {"output_tokens": 1}},
        )
        m = _run(_replay_one_request(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            url="http://test",
            model="test",
            headers={},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            seq=1,
            timeout=30.0,
            upstream_api="anthropic",
            response_chunks=None,
        ))
        assert m.error is None


# =========================================================================
# _replay_task_entries_live -- full integration
# =========================================================================


def _make_fake_response(text: str) -> FakeStreamResponse:
    return FakeStreamResponse(_make_sse_lines([_sse_chunk(text)]))


class TestReplayTaskEntriesLive:
    def test_multi_turn_uses_actual_responses(self):
        """The core fix: turn 2's messages should contain the server's actual
        response from turn 1, not the recorded one."""
        entries = _make_growing_entries(3)

        client = SequentialFakeClient([
            _make_fake_response("Server answer 1"),
            _make_fake_response("Server answer 2"),
            _make_fake_response("Server answer 3"),
        ])

        results = _run(_replay_task_entries_live(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
        ))

        assert len(results) == 3

        # Turn 1: [system, user1]
        turn1_msgs = client.captured_messages[0]
        assert len(turn1_msgs) == 2
        assert turn1_msgs[0]["role"] == "system"
        assert turn1_msgs[1]["content"] == "Question 1"

        # Turn 2: [system, user1, SERVER_RESPONSE_1, user2]
        turn2_msgs = client.captured_messages[1]
        assert len(turn2_msgs) == 4
        assert turn2_msgs[2]["role"] == "assistant"
        assert turn2_msgs[2]["content"] == "Server answer 1"
        assert "Recorded" not in turn2_msgs[2]["content"]
        assert turn2_msgs[3]["content"] == "Question 2"

        # Turn 3: [system, user1, SERVER_1, user2, SERVER_2, user3]
        turn3_msgs = client.captured_messages[2]
        assert len(turn3_msgs) == 6
        assert turn3_msgs[4]["content"] == "Server answer 2"
        assert turn3_msgs[5]["content"] == "Question 3"

    def test_single_entry_works_like_recorded(self):
        """Single-entry tasks should behave identically to the old code."""
        entries = _make_growing_entries(1)
        client = SequentialFakeClient([_make_fake_response("answer")])

        results = _run(_replay_task_entries_live(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
        ))

        assert len(results) == 1
        msgs = client.captured_messages[0]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_error_falls_back_to_recorded_assistant(self):
        """On error, the recorded assistant message should be used as fallback
        so subsequent entries can still proceed."""
        entries = _make_growing_entries(3)

        error_resp = FakeStreamResponse([], status_code=500, body=b"Server Error")
        client = SequentialFakeClient([
            _make_fake_response("Server answer 1"),
            error_resp,
            _make_fake_response("Server answer 3"),
        ])

        results = _run(_replay_task_entries_live(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
        ))

        assert len(results) == 3
        assert results[1].error is not None

        # Turn 3 should still proceed, using the recorded fallback for turn 2.
        # The fallback is "Recorded answer 1" -- the assistant message from the
        # delta between entry[0] and entry[1] (the recorded response to turn 1).
        turn3_msgs = client.captured_messages[2]
        assert turn3_msgs[4]["role"] == "assistant"
        assert turn3_msgs[4]["content"] == "Recorded answer 1"
        assert turn3_msgs[5]["content"] == "Question 3"

    def test_model_context_length_skip(self):
        """Entries exceeding model_context_length should be skipped without
        corrupting the live history."""
        entries = _make_growing_entries(2)

        client = SequentialFakeClient([
            _make_fake_response("answer 1"),
            _make_fake_response("answer 2"),
        ])

        results = _run(_replay_task_entries_live(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
            model_context_length=1,
        ))

        # Both entries exceed 1 token so both should be skipped
        assert len(results) == 0

    def test_on_complete_callback_called(self):
        entries = _make_growing_entries(2)
        client = SequentialFakeClient([
            _make_fake_response("a1"),
            _make_fake_response("a2"),
        ])

        call_count = 0

        def on_complete():
            nonlocal call_count
            call_count += 1

        _run(_replay_task_entries_live(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
            on_complete=on_complete,
        ))

        assert call_count == 2


# =========================================================================
# _replay_task_entries (recorded mode) -- backward compat
# =========================================================================


class TestReplayTaskEntriesRecorded:
    def test_sends_recorded_messages_verbatim(self):
        """The old code path sends entry.messages as-is, including recorded assistants."""
        entries = _make_growing_entries(2)
        client = SequentialFakeClient([
            _make_fake_response("ignored 1"),
            _make_fake_response("ignored 2"),
        ])

        _run(_replay_task_entries(
            client=client,
            url="http://test/v1/chat/completions",
            model_override="test",
            headers={},
            entries=entries,
            timeout=30.0,
            user_id=0,
        ))

        # Turn 2 should have the RECORDED assistant, not the server's response
        turn2_msgs = client.captured_messages[1]
        assert turn2_msgs[2]["content"] == "Recorded answer 1"
