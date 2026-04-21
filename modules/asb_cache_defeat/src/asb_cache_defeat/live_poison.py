"""Prefix-cache poisoning for realistic KV cache invalidation.

Simulates how real agentic sessions break the KV prefix cache: when an
agent edits files mid-conversation, the context changes from the edit
point onward, invalidating the cache for all subsequent tokens.

This module reproduces that pattern by finding isolated single spaces
after the common prefix and randomly doubling some of them. Doubling a
space inserts an extra token (BPE tokenizers encode " word" as one token,
but "  word" becomes two), which shifts all subsequent tokens and breaks
the prefix cache from that point.

Algorithm:
  1. Compute the LCP across all tasks in the scenario (the shared system
     prompt). Everything before len(LCP) stays cached and untouched.
  2. For each (task, repetition) execution, generate a unique 64-bit
     poisonset seeded by (task_id, execution_index).
  3. Apply the SAME poisonset to ALL requests in one execution: find
     isolated spaces after the LCP, double them where mask[i] == 1.
  4. Different executions of the same task get different poisonsets,
     breaking the prefix cache between repetitions.
"""

from __future__ import annotations

import random

from agentic_swarm_bench.scenarios.registry import RecordingEntry, Task

MAX_POSITIONS = 64


def compute_lcp(strings: list[str]) -> str:
    """Compute the longest common prefix of a list of strings."""
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]

    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        if any(s[i] != ch for s in strings):
            return shortest[:i]
    return shortest


def generate_poison_mask(seed: str, n_positions: int = MAX_POSITIONS) -> list[int]:
    """Generate a deterministic binary mask from an arbitrary seed string."""
    rng = random.Random(f"poison-{seed}")
    return [rng.randint(0, 1) for _ in range(n_positions)]


def find_isolated_spaces(text: str, start: int, max_count: int = MAX_POSITIONS) -> list[int]:
    """Find positions of isolated single spaces after start index.

    An isolated space is a space character where neither the preceding
    nor the following character is also a space. These are safe to double
    without creating triple-space runs.
    """
    positions: list[int] = []
    text_len = len(text)

    for i in range(max(start, 0), text_len):
        if text[i] != " ":
            continue
        prev_is_space = i > 0 and text[i - 1] == " "
        next_is_space = i < text_len - 1 and text[i + 1] == " "
        if prev_is_space or next_is_space:
            continue
        positions.append(i)
        if len(positions) >= max_count:
            break

    return positions


def _serialize_messages(messages: list[dict]) -> str:
    """Concatenate all message content into one string."""
    return "".join(m.get("content", "") for m in messages)


def _apply_space_doubling(text: str, positions: list[int], mask: list[int]) -> str:
    """Double spaces at selected positions according to the mask.

    Processes positions from right to left so insertions don't shift
    indices of subsequent positions.
    """
    n = min(len(positions), len(mask))
    if n == 0:
        return text

    chars = list(text)
    for i in range(n - 1, -1, -1):
        if mask[i] == 1:
            chars.insert(positions[i] + 1, " ")

    return "".join(chars)


def _reconstruct_messages(
    original_messages: list[dict],
    original_serialized: str,
    poisoned_serialized: str,
) -> list[dict]:
    """Map poisoned text back onto original message boundaries.

    Walks through the original messages, tracking how the poisoned text
    has expanded due to space insertions, and rebuilds each message's
    content from the poisoned string.
    """
    if original_serialized == poisoned_serialized:
        return original_messages

    result: list[dict] = []
    orig_offset = 0
    poison_offset = 0

    for msg in original_messages:
        content = msg.get("content", "")
        content_len = len(content)

        if content_len == 0:
            result.append(dict(msg))
            continue

        orig_end = orig_offset + content_len

        poison_end = poison_offset
        orig_scan = orig_offset
        while orig_scan < orig_end:
            if poison_end >= len(poisoned_serialized):
                break
            if (
                poisoned_serialized[poison_end] == " "
                and poison_end + 1 < len(poisoned_serialized)
                and poisoned_serialized[poison_end + 1] == " "
                and original_serialized[orig_scan] == " "
                and (orig_scan == 0 or original_serialized[orig_scan - 1] != " ")
                and (
                    orig_scan + 1 >= len(original_serialized)
                    or original_serialized[orig_scan + 1] != " "
                )
            ):
                poison_end += 2
                orig_scan += 1
            else:
                poison_end += 1
                orig_scan += 1

        new_content = poisoned_serialized[poison_offset:poison_end]
        result.append({**msg, "content": new_content})

        orig_offset = orig_end
        poison_offset = poison_end

    return result


def compute_scenario_lcp(tasks: list[Task]) -> int:
    """Compute the LCP length across all tasks in a scenario.

    Serializes the first request of each task and finds their common
    prefix. This is typically the shared system prompt / agent prompt
    that all tasks start with.
    """
    if not tasks:
        return 0

    first_request_texts: list[str] = []
    for task in tasks:
        if not task.entries:
            return 0
        first_request_texts.append(_serialize_messages(task.entries[0].messages))

    if not first_request_texts:
        return 0

    return len(compute_lcp(first_request_texts))


def poison_task_execution(
    task: Task,
    lcp_len: int,
    execution_index: int,
) -> Task:
    """Poison a single execution of a task.

    The same mask is applied to ALL requests in this execution. Different
    execution_index values produce different masks, so repetitions of the
    same task break the prefix cache differently.

    Args:
        task: The task to poison.
        lcp_len: Length of the cross-task LCP (from compute_scenario_lcp).
        execution_index: Which repetition this is (0, 1, 2, ...).

    Returns:
        A new Task with poisoned entries (original is not modified).
    """
    if not task.entries:
        return task

    mask = generate_poison_mask(f"{task.id}-exec-{execution_index}")

    new_entries: list[RecordingEntry] = []
    for entry in task.entries:
        text = _serialize_messages(entry.messages)
        positions = find_isolated_spaces(text, lcp_len)

        if not positions:
            new_entries.append(entry)
            continue

        poisoned_text = _apply_space_doubling(text, positions, mask)

        if poisoned_text == text:
            new_entries.append(entry)
            continue

        new_messages = _reconstruct_messages(entry.messages, text, poisoned_text)

        new_entry = RecordingEntry(
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
        )
        new_entries.append(new_entry)

    return Task(
        id=task.id,
        name=task.name,
        entries=new_entries,
    )
