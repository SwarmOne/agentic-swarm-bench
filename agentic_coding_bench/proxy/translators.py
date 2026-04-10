"""Translate between Anthropic Messages API and OpenAI Chat Completions API."""

from __future__ import annotations

import json
import uuid


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
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") in ("tool_result", "tool_use"):
                    text_parts.append(json.dumps(block))
            messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            messages.append({"role": role, "content": str(content) if content else ""})

    oai = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
        "temperature": body.get("temperature", 1.0),
    }

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

    return {
        "id": "msg_" + oai_resp.get("id", uuid.uuid4().hex[:24]),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": message.get("content", "")}],
        "stop_reason": (
            "end_turn" if choice.get("finish_reason") == "stop" else choice.get("finish_reason")
        ),
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
