"""Tests for proxy translation and padding."""

from agentic_swarm_bench.proxy.padding import (
    count_tokens_approx,
    pad_messages_to_target,
    poison_messages,
    poison_text_spaces,
)
from agentic_swarm_bench.proxy.translators import anthropic_to_openai, openai_to_anthropic_response


def test_anthropic_to_openai_basic():
    body = {
        "model": "claude-3-opus",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 1024,
        "stream": True,
        "temperature": 0.5,
    }
    result = anthropic_to_openai(body, model="my-oai-model")
    assert result["model"] == "my-oai-model"
    assert result["stream"] is True
    assert result["max_tokens"] == 1024
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "You are helpful."
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][1]["content"] == "Hello"


def test_anthropic_to_openai_system_list():
    body = {
        "model": "claude-3-opus",
        "system": [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ],
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 512,
    }
    result = anthropic_to_openai(body, model="test")
    assert "Part one." in result["messages"][0]["content"]
    assert "Part two." in result["messages"][0]["content"]


def test_openai_to_anthropic_response():
    oai = {
        "id": "chatcmpl-test123",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello back!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }
    result = openai_to_anthropic_response(oai, "claude-3")
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["content"][0]["text"] == "Hello back!"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 5


def test_count_tokens_approx():
    assert count_tokens_approx("hello world") == 2  # 11 chars / 4
    assert count_tokens_approx("a" * 400) == 100


def test_pad_messages_adds_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    padded = pad_messages_to_target(messages, target_tokens=5000, defeat_cache=False)
    total_chars = sum(len(m["content"]) for m in padded)
    assert total_chars > 1000


def test_pad_messages_with_cache_defeat():
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Hello"},
    ]
    padded_a = pad_messages_to_target(messages, target_tokens=5000, defeat_cache=True)
    padded_b = pad_messages_to_target(messages, target_tokens=5000, defeat_cache=True)
    assert padded_a[0]["content"] != padded_b[0]["content"]
    assert "[session_id=" not in padded_a[0]["content"]


def test_pad_messages_zero_target():
    messages = [{"role": "user", "content": "Hi"}]
    result = pad_messages_to_target(messages, target_tokens=0)
    assert result == messages


def test_poison_text_spaces_deterministic():
    text = "hello world this is a test"
    a = poison_text_spaces(text, seed="same")
    b = poison_text_spaces(text, seed="same")
    assert a == b


def test_poison_text_spaces_different_seeds():
    text = "hello world this is a test with many spaces in it for poisoning"
    a = poison_text_spaces(text, seed="seed-1")
    b = poison_text_spaces(text, seed="seed-2")
    assert a != b


def test_poison_text_spaces_doubles_spaces():
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    poisoned = poison_text_spaces(text, seed="test")
    assert "  " in poisoned
    assert len(poisoned) > len(text)


def test_poison_messages_returns_new_list():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there friend."},
    ]
    poisoned = poison_messages(msgs, seed="test")
    assert len(poisoned) == 2
    assert poisoned is not msgs
    assert poisoned[0] is not msgs[0]


# ---------------------------------------------------------------------------
# Translator: tool_use and tool_result blocks
# ---------------------------------------------------------------------------


def test_tool_use_block_serialized():
    body = {
        "model": "claude-3",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "Read",
                        "input": {"path": "/foo.py"},
                    }
                ],
            }
        ],
        "max_tokens": 512,
    }
    result = anthropic_to_openai(body, model="test")
    assistant_msg = next(m for m in result["messages"] if m["role"] == "assistant")

    assert "tool_calls" in assistant_msg
    tc = assistant_msg["tool_calls"][0]
    assert tc["function"]["name"] == "Read"
    import json

    assert json.loads(tc["function"]["arguments"]) == {"path": "/foo.py"}


def test_tool_result_block_serialized():
    body = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "file contents here",
                    }
                ],
            }
        ],
        "max_tokens": 512,
    }
    result = anthropic_to_openai(body, model="test")
    tool_msg = next(m for m in result["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "tool_123"
    assert tool_msg["content"] == "file contents here"


def test_top_p_passed_through():
    body = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_p": 0.9,
    }
    result = anthropic_to_openai(body, model="test")
    assert result["top_p"] == 0.9


def test_top_p_absent_when_not_set():
    body = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }
    result = anthropic_to_openai(body, model="test")
    assert "top_p" not in result


def test_mixed_content_list_concatenated():
    """Multiple text blocks in content list are joined."""
    body = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ],
            }
        ],
        "max_tokens": 100,
    }
    result = anthropic_to_openai(body, model="test")
    user_msg = next(m for m in result["messages"] if m["role"] == "user")
    assert "Hello" in user_msg["content"]
    assert "world" in user_msg["content"]
