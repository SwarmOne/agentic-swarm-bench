"""Tests for context padding and cache defeat."""

from agentic_swarm_bench.tasks.context.codebase_context import (
    build_context_block,
    build_messages,
)


def test_build_context_block_length():
    block = build_context_block(10000)
    assert len(block) == 10000


def test_build_context_block_contains_code():
    block = build_context_block(50000)
    assert "def " in block or "function" in block.lower() or "var_" in block
    assert "tool" in block.lower()


def test_build_messages_structure():
    msgs = build_messages("Write hello world", target_tokens=1000)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "hello world" in msgs[1]["content"].lower()


def test_build_messages_no_salt():
    """Messages should never contain salt injection."""
    msgs = build_messages("Write hello world", target_tokens=1000)
    user_content = msgs[1]["content"]
    assert "[session_id=" not in user_content


def test_build_messages_approximate_size():
    msgs = build_messages("Write hello world", target_tokens=20000)
    total_chars = sum(len(m["content"]) for m in msgs)
    total_tokens_approx = total_chars // 4
    assert total_tokens_approx >= 15000
