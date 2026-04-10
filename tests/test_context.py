"""Tests for context padding and cache defeat."""

from agentic_coding_bench.tasks.context.codebase_context import (
    build_cache_defeat_salt,
    build_context_block,
    build_messages,
)


def test_cache_defeat_salt_is_unique():
    salts = {build_cache_defeat_salt() for _ in range(20)}
    assert len(salts) == 20


def test_cache_defeat_salt_format():
    salt = build_cache_defeat_salt()
    assert salt.startswith("[session_id=")
    assert "ts=" in salt
    assert "rand=" in salt


def test_build_context_block_length():
    block = build_context_block(10000)
    assert len(block) == 10000


def test_build_context_block_contains_code():
    block = build_context_block(50000)
    assert "def " in block or "function" in block.lower() or "var_" in block
    assert "tool" in block.lower()


def test_build_messages_structure():
    msgs = build_messages("Write hello world", target_tokens=1000, defeat_cache=False)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "hello world" in msgs[1]["content"].lower()


def test_build_messages_with_cache_defeat():
    msgs = build_messages("Write hello world", target_tokens=1000, defeat_cache=True)
    user_content = msgs[1]["content"]
    assert "[session_id=" in user_content


def test_build_messages_without_cache_defeat():
    msgs = build_messages("Write hello world", target_tokens=1000, defeat_cache=False)
    user_content = msgs[1]["content"]
    assert "[session_id=" not in user_content


def test_build_messages_approximate_size():
    msgs = build_messages("Write hello world", target_tokens=20000, defeat_cache=False)
    total_chars = sum(len(m["content"]) for m in msgs)
    total_tokens_approx = total_chars // 4
    assert total_tokens_approx >= 15000
