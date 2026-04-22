"""Replay support utilities for scenario tasks.

Provides compute_scenario_lcp() for longest-common-prefix validation
and poison_task_execution() as a hook for cache-defeat processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_swarm_bench.scenarios.registry import Task

_HAS_CACHE_DEFEAT = False


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
    is shared across tasks. Used for ``min_lcp_length`` validation.
    """
    all_texts: list[str] = []
    for task in tasks:
        for entry in task.entries:
            all_texts.append(_flatten_message_text(entry.messages))
    return _compute_lcp_of_strings(all_texts)




def poison_task_execution(
    task: Task,
    lcp_len: int,
    execution_index: int,
) -> Task:
    """Return a copy of *task*, possibly with text varied for cache defeat.

    Built-in scenarios ship with pre-poisoned recordings; this hook
    applies any additional processing needed per execution.
    """
    return task
