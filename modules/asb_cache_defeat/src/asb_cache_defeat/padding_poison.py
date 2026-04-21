"""Space-doubling poison for the speed benchmark (asb speed).

This is the per-message variant used by runner/direct.py. The scenario-level
variant lives in live_poison.py.
"""

from __future__ import annotations

import random
import time


def poison_text_spaces(text: str, seed: str, max_positions: int = 64) -> str:
    """Defeat prefix caching by randomly doubling isolated spaces.

    Finds isolated single spaces (not adjacent to other spaces) and
    randomly doubles some of them. This shifts BPE token boundaries
    and invalidates the KV cache without adding artificial content.
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
