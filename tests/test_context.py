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


# ---------------------------------------------------------------------------
# Exact length guarantee for build_context_block
# ---------------------------------------------------------------------------


def test_build_context_block_exact_length():
    for n in [100, 1000, 5000, 10000]:
        block = build_context_block(n)
        assert len(block) == n, f"Expected {n} chars, got {len(block)}"


# ---------------------------------------------------------------------------
# Determinism: same args → same result (no random salt)
# ---------------------------------------------------------------------------


def test_build_messages_deterministic():
    a = build_messages("Write a function", target_tokens=5000)
    b = build_messages("Write a function", target_tokens=5000)
    assert a[0]["content"] == b[0]["content"]
    assert a[1]["content"] == b[1]["content"]


def test_build_context_block_deterministic():
    a = build_context_block(5000)
    b = build_context_block(5000)
    assert a == b


# ---------------------------------------------------------------------------
# Edge cases: very small target_tokens
# ---------------------------------------------------------------------------


def test_build_messages_tiny_tokens_no_crash():
    msgs = build_messages("hi", target_tokens=1)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "hi" in msgs[1]["content"].lower()


def test_build_messages_zero_tokens_no_crash():
    msgs = build_messages("hi", target_tokens=0)
    assert len(msgs) == 2
