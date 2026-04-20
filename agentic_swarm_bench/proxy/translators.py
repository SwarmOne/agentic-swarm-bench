"""Translate between Anthropic Messages API and OpenAI Chat Completions API."""

from __future__ import annotations

import json
import uuid


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    oai_tools = []
    for tool in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return oai_tools


def _anthropic_tool_choice_to_openai(tc: dict | str) -> str | dict | None:
    """Convert Anthropic tool_choice to OpenAI tool_choice."""
    if isinstance(tc, str):
        return tc
    tc_type = tc.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


def _convert_anthropic_message(msg: dict) -> list[dict]:
    """Convert one Anthropic message to one or more OpenAI messages.

    An Anthropic assistant message with tool_use blocks becomes an OpenAI
    assistant message with tool_calls. Anthropic user messages containing
    tool_result blocks become separate OpenAI tool-role messages.
    """
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": str(content) if content else ""}]

    if role == "assistant":
        text_parts = []
        tool_calls = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", "call_" + uuid.uuid4().hex[:8]),
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        oai_msg: dict = {"role": "assistant"}
        oai_msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            oai_msg["tool_calls"] = tool_calls
        return [oai_msg]

    # User messages: extract tool_result blocks as separate tool messages
    tool_msgs: list[dict] = []
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_content = "\n".join(
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": str(result_content),
            })
        else:
            text_parts.append(json.dumps(block))

    result: list[dict] = []
    if text_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    result.extend(tool_msgs)
    if not result:
        result.append({"role": "user", "content": ""})
    return result


def anthropic_to_openai(body: dict, model: str) -> dict:
    """Convert an Anthropic /v1/messages request to OpenAI /v1/chat/completions."""
    messages = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [b.get("text", "") for b in system if b.get("type") == "text"]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in body.get("messages", []):
        messages.extend(_convert_anthropic_message(msg))

    oai: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
        "temperature": body.get("temperature", 1.0),
    }

    if body.get("tools"):
        oai["tools"] = _anthropic_tools_to_openai(body["tools"])
    if body.get("tool_choice"):
        oai["tool_choice"] = _anthropic_tool_choice_to_openai(body["tool_choice"])

    if body.get("top_p") is not None:
        oai["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]
    if oai["stream"]:
        oai["stream_options"] = {"include_usage": True}

    return oai


def openai_to_anthropic_response(oai_resp: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI response to Anthropic format."""
    choice = oai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = oai_resp.get("usage", {})

    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            tool_input = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {"raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", "toolu_" + uuid.uuid4().hex[:8]),
            "name": fn.get("name", ""),
            "input": tool_input,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason", "stop")
    stop_reason = "tool_use" if finish == "tool_calls" else (
        "end_turn" if finish == "stop" else finish
    )

    return {
        "id": "msg_" + oai_resp.get("id", uuid.uuid4().hex[:24]),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def make_anthropic_stream_events(model: str, msg_id: str) -> list[str]:
    """Return the opening SSE events for an Anthropic streaming response."""
    start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    return [
        f"event: message_start\ndata: {json.dumps(start)}\n\n",
        f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n",
    ]


class StreamingToolCallAccumulator:
    """Accumulate streaming OpenAI tool_calls deltas, emit Anthropic SSE events."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}
        self._block_index = 1  # 0 is the text block

    def process_chunk(self, chunk: dict) -> list[str]:
        """Process one OpenAI streaming chunk, return Anthropic SSE event strings."""
        events: list[str] = []
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                fn = tc_delta.get("function", {})

                if idx not in self._calls:
                    self._calls[idx] = {
                        "id": tc_delta.get("id", "toolu_" + uuid.uuid4().hex[:8]),
                        "name": fn.get("name", ""),
                        "arguments": "",
                        "block_index": self._block_index,
                    }
                    self._block_index += 1
                    events.append(self._block_start_event(self._calls[idx]))
                else:
                    if fn.get("name"):
                        self._calls[idx]["name"] = fn["name"]

                arg_chunk = fn.get("arguments", "")
                if arg_chunk:
                    self._calls[idx]["arguments"] += arg_chunk
                    events.append(self._input_delta_event(
                        self._calls[idx]["block_index"], arg_chunk
                    ))

            finish = choice.get("finish_reason")
            if finish == "tool_calls":
                for call_info in self._calls.values():
                    events.append(self._block_stop_event(call_info["block_index"]))

        return events

    @staticmethod
    def _block_start_event(call: dict) -> str:
        evt = {
            "type": "content_block_start",
            "index": call["block_index"],
            "content_block": {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
            },
        }
        return f"event: content_block_start\ndata: {json.dumps(evt)}\n\n"

    @staticmethod
    def _input_delta_event(block_index: int, partial: str) -> str:
        evt = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        }
        return f"event: content_block_delta\ndata: {json.dumps(evt)}\n\n"

    @staticmethod
    def _block_stop_event(block_index: int) -> str:
        evt = {"type": "content_block_stop", "index": block_index}
        return f"event: content_block_stop\ndata: {json.dumps(evt)}\n\n"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._calls)
