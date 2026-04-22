"""Prefix-cache poisoning for replay scenarios.

Provides compute_scenario_lcp() and poison_task_execution() used by
the replay engine to compute the longest common prefix across tasks
and apply text variation that defeats KV-cache prefix matching.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_swarm_bench.scenarios.registry import Task

from agentic_swarm_bench.scenarios.registry import RecordingEntry
from agentic_swarm_bench.scenarios.registry import Task as TaskCls

_HAS_CACHE_DEFEAT = False
# --- PRIVATE ---
try:
    from asb_cache_defeat.live_poison import (
        compute_scenario_lcp as _real_compute_scenario_lcp,
    )
    from asb_cache_defeat.live_poison import (
        poison_task_execution as _real_poison_task_execution,
    )

    _HAS_CACHE_DEFEAT = True
except ImportError:
    pass
# --- /PRIVATE ---


def _flatten_message_text(messages: list[dict]) -> str:
    """Concatenate all message content into a single string for LCP comparison."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", "") or block.get("thinking", ""))
        parts.append(msg.get("reasoning_content", ""))
    return "".join(parts)


def _compute_lcp_of_strings(strings: list[str]) -> int:
    """Return the length of the longest common prefix across all strings."""
    if not strings:
        return 0
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        for s in strings:
            if s[i] != ch:
                return i
    return len(shortest)


def compute_scenario_lcp(tasks: list[Task]) -> int:
    """Compute the longest common prefix (in chars) across all task entries.

    Uses character-level comparison to determine how much of the prompt
    is shared across tasks. This value drives the ``realistic`` cache
    mode: content within the LCP is preserved (cacheable); content
    beyond it is varied per execution to defeat caching.
    """
    # --- PRIVATE ---
    if _HAS_CACHE_DEFEAT:
        return _real_compute_scenario_lcp(tasks)
    # --- /PRIVATE ---
    all_texts: list[str] = []
    for task in tasks:
        for entry in task.entries:
            all_texts.append(_flatten_message_text(entry.messages))
    return _compute_lcp_of_strings(all_texts)


def _poison_text(text: str, seed: str) -> str:
    """Apply text variation to defeat KV-cache prefix matching.

    Inserts deterministic variation (extra spaces, punctuation swaps)
    seeded by (task_id, execution_index) so each execution sends
    different bytes. A unique hash suffix is appended to guarantee
    no two executions produce identical content.
    """
    if not text:
        return text

    rng = random.Random(seed)

    chars = list(text)
    length = len(chars)
    mutations = max(1, length // 200)

    for _ in range(mutations):
        pos = rng.randint(0, length - 1)
        ch = chars[pos]
        if ch == " ":
            chars[pos] = "  "
        elif ch == ".":
            chars[pos] = rng.choice([".", ". "])
        elif ch == ",":
            chars[pos] = rng.choice([",", ", "])
        elif ch == "\n":
            chars[pos] = "\n "

    tag = hashlib.sha256(seed.encode()).hexdigest()[:8]
    suffix = f"\n<!-- {tag} -->"
    return "".join(chars) + suffix


def poison_task_execution(
    task: Task,
    lcp_len: int,
    execution_index: int,
) -> Task:
    """Return a copy of *task* with messages varied to defeat the KV cache.

    Applies deterministic text mutations seeded by (task.id,
    execution_index) so each execution sends different bytes.

    When *lcp_len* > 0 (realistic mode), only content beyond the shared
    prefix is mutated. When *lcp_len* == 0 (allcold mode), everything is
    mutated.
    """
    # --- PRIVATE ---
    if _HAS_CACHE_DEFEAT:
        return _real_poison_task_execution(task, lcp_len, execution_index)
    # --- /PRIVATE ---
    poisoned_entries: list[RecordingEntry] = []
    for entry in task.entries:
        new_messages: list[dict] = []
        cumulative_len = 0
        for msg in entry.messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content", "")

            if isinstance(content, str) and content:
                before_len = cumulative_len
                cumulative_len += len(content)
                if cumulative_len > lcp_len:
                    poison_start = max(0, lcp_len - before_len)
                    safe_prefix = content[:poison_start]
                    to_poison = content[poison_start:]
                    seed = f"{task.id}-{execution_index}-{entry.seq}-{before_len}"
                    msg_copy["content"] = safe_prefix + _poison_text(to_poison, seed)
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if not isinstance(block, dict):
                        new_blocks.append(block)
                        continue
                    block_copy = dict(block)
                    for key in ("text", "thinking"):
                        text = block_copy.get(key, "")
                        if text:
                            before_len = cumulative_len
                            cumulative_len += len(text)
                            if cumulative_len > lcp_len:
                                poison_start = max(0, lcp_len - before_len)
                                safe = text[:poison_start]
                                seed = f"{task.id}-{execution_index}-{entry.seq}-{key}-{before_len}"
                                block_copy[key] = safe + _poison_text(text[poison_start:], seed)
                    new_blocks.append(block_copy)
                msg_copy["content"] = new_blocks

            reasoning = msg_copy.get("reasoning_content", "")
            if reasoning:
                before_len = cumulative_len
                cumulative_len += len(reasoning)
                if cumulative_len > lcp_len:
                    poison_start = max(0, lcp_len - before_len)
                    safe = reasoning[:poison_start]
                    seed = f"{task.id}-{execution_index}-{entry.seq}-reasoning-{before_len}"
                    poisoned = _poison_text(reasoning[poison_start:], seed)
                    msg_copy["reasoning_content"] = safe + poisoned

            new_messages.append(msg_copy)

        poisoned_entries.append(RecordingEntry(
            seq=entry.seq,
            experiment_id=entry.experiment_id,
            timestamp=entry.timestamp,
            messages=new_messages,
            model=entry.model,
            max_tokens=entry.max_tokens,
            temperature=entry.temperature,
            stream=entry.stream,
            prompt_tokens=entry.prompt_tokens,
            ttft_ms=entry.ttft_ms,
            total_time_s=entry.total_time_s,
            completion_tokens=entry.completion_tokens,
            tok_per_sec=entry.tok_per_sec,
        ))

    return TaskCls(
        id=task.id,
        name=task.name,
        entries=poisoned_entries,
        evaluate=task.evaluate,
    )
