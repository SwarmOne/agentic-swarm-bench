"""Context padding for the recording proxy.

Pads messages to a target token count with realistic coding context,
optionally injecting a unique salt to defeat prefix caching.
"""

from __future__ import annotations

from agentic_coding_bench.tasks.context.codebase_context import (
    build_cache_defeat_salt,
    build_context_block,
)


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def pad_messages_to_target(
    messages: list[dict],
    target_tokens: int,
    defeat_cache: bool = True,
) -> list[dict]:
    """Inject context padding into a message list to reach target token count.

    Prepends padding to the existing system message (or inserts one) so
    the message structure stays valid for the downstream model.
    """
    if target_tokens <= 0:
        return messages

    current_chars = sum(len(m.get("content", "")) for m in messages)
    current_tokens = current_chars // 4
    if current_tokens >= target_tokens:
        return messages

    salt = ""
    if defeat_cache:
        salt = build_cache_defeat_salt() + "\n"

    needed_chars = (target_tokens - current_tokens) * 4 - len(salt)
    if needed_chars <= 0:
        return messages

    padding = salt + build_context_block(needed_chars)

    result = [dict(m) for m in messages]
    for i, m in enumerate(result):
        if m["role"] == "system":
            result[i] = {**m, "content": padding + "\n\n" + m["content"]}
            return result

    result.insert(0, {"role": "system", "content": padding})
    return result
