"""Context padding and prefix-cache poisoning utilities.

Provides:
- Message padding: fills messages to a target token count with realistic context
- Space-doubling poison: defeats prefix caching by randomly doubling isolated
  spaces, shifting BPE token boundaries without adding artificial content
"""

from __future__ import annotations

import random
import time

from agentic_swarm_bench.tasks.context.codebase_context import build_context_block


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def poison_text_spaces(text: str, seed: str, max_positions: int = 64) -> str:
    """Defeat prefix caching by randomly doubling isolated spaces.

    Finds isolated single spaces (not adjacent to other spaces) and
    randomly doubles some of them. This shifts BPE token boundaries
    and invalidates the KV cache without adding artificial content.

    Same technique as scenarios/poison.py, adapted for single-request use.
    """
    rng = random.Random(seed)
    chars = list(text)
    text_len = len(chars)

    doubled = 0
    for i in range(text_len - 1, -1, -1):
        if chars[i] != " ":
            continue
        prev_is_space = i > 0 and chars[i - 1] == " "
        next_is_space = i < text_len - 1 and chars[i + 1] == " "
        if prev_is_space or next_is_space:
            continue
        if rng.random() < 0.5:
            chars.insert(i + 1, " ")
            doubled += 1
        if doubled >= max_positions:
            break

    return "".join(chars)


def poison_messages(messages: list[dict], seed: str | None = None) -> list[dict]:
    """Apply space-doubling poison to all message content.

    If no seed is provided, generates one from the current timestamp.
    Returns a new list of messages (originals are not modified).
    """
    if seed is None:
        seed = f"poison-{time.time_ns()}"

    result = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            content = poison_text_spaces(content, seed)
        result.append({**msg, "content": content})
    return result


def pad_messages_to_target(
    messages: list[dict],
    target_tokens: int,
    defeat_cache: bool = True,
) -> list[dict]:
    """Inject context padding into a message list to reach target token count.

    Prepends padding to the existing system message (or inserts one) so
    the message structure stays valid for the downstream model.

    When defeat_cache is True, applies space-doubling to the padding
    to break prefix caching.
    """
    if target_tokens <= 0:
        return messages

    current_chars = sum(len(m.get("content", "")) for m in messages)
    current_tokens = current_chars // 4
    if current_tokens >= target_tokens:
        return messages

    needed_chars = (target_tokens - current_tokens) * 4
    if needed_chars <= 0:
        return messages

    padding = build_context_block(needed_chars)

    if defeat_cache:
        seed = f"proxy-{time.time_ns()}"
        padding = poison_text_spaces(padding, seed)

    result = [dict(m) for m in messages]
    for i, m in enumerate(result):
        if m["role"] == "system":
            result[i] = {**m, "content": padding + "\n\n" + m["content"]}
            return result

    result.insert(0, {"role": "system", "content": padding})
    return result
