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
