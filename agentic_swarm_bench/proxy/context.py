"""Context padding utilities for the proxy server.

Provides:
- Token counting approximation
- Message padding to a target token count using realistic context
"""

from __future__ import annotations

from agentic_swarm_bench.tasks.context.codebase_context import build_context_block


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

    When defeat_cache is True and a cache-defeat extension is installed,
    applies token-boundary perturbation to the padding to break prefix
    caching. Otherwise padding is inserted verbatim.
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

    # --- PRIVATE ---
    if defeat_cache:
        try:
            import time

            from asb_cache_defeat.padding_poison import poison_text_spaces

            seed = f"proxy-{time.time_ns()}"
            padding = poison_text_spaces(padding, seed)
        except ImportError:
            pass
    # --- /PRIVATE ---

    result = [dict(m) for m in messages]
    for i, m in enumerate(result):
        if m["role"] == "system":
            result[i] = {**m, "content": padding + "\n\n" + m["content"]}
            return result

    result.insert(0, {"role": "system", "content": padding})
    return result
