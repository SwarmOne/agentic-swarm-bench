"""Tests for the asb_cache_defeat module.

Covers:
  - LCP computation
  - Poison mask generation
  - Isolated space finder
  - Scenario LCP
  - Full task execution poisoning
  - Per-message space-doubling (padding_poison)
  - Proxy padding with cache defeat
  - Player repetition poisoning invariants
  - Report methodology section
"""

from __future__ import annotations

from asb_cache_defeat.live_poison import (
    _serialize_messages,
    compute_lcp,
    compute_scenario_lcp,
    find_isolated_spaces,
    generate_poison_mask,
    poison_task_execution,
)
from asb_cache_defeat.padding_poison import poison_messages, poison_text_spaces

from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_swarm_bench.proxy.context import pad_messages_to_target
from agentic_swarm_bench.report.markdown import generate_report
from agentic_swarm_bench.scenarios.registry import RecordingEntry, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_growing_task(
    task_id="growing-task",
    n_requests=5,
    system_prompt="You are a coding assistant.",
):
    """Create a task with growing messages (like a real agentic conversation)."""
    entries = []
    conversation = [{"role": "system", "content": system_prompt}]

    for i in range(n_requests):
        user_msg = f"Please fix function_{i} in the codebase"
        conversation = conversation + [{"role": "user", "content": user_msg}]

        entries.append(
            RecordingEntry(
                seq=i + 1,
                experiment_id="test",
                messages=[dict(m) for m in conversation],
            )
        )

        assistant_msg = f"I fixed function_{i} by updating the logic"
        conversation = conversation + [{"role": "assistant", "content": assistant_msg}]

    return Task(id=task_id, name=f"Task {task_id}", entries=entries)


def _make_poisonable_entry(content: str) -> RecordingEntry:
    return RecordingEntry(seq=0, messages=[{"role": "user", "content": content}])


def _make_poisonable_tasks() -> list[Task]:
    """Two tasks sharing a prefix (so compute_scenario_lcp returns a finite LCP)."""
    shared = "SYSTEM PROMPT " * 50
    body_a = " ".join(f"alpha{i}" for i in range(400))
    body_b = " ".join(f"beta{i}" for i in range(400))
    return [
        Task(id="t1", name="t1", entries=[_make_poisonable_entry(shared + body_a)]),
        Task(id="t2", name="t2", entries=[_make_poisonable_entry(shared + body_b)]),
    ]


def _make_run(model="test-model", endpoint="http://test:8000"):
    run = BenchmarkRun(
        model=model,
        endpoint=endpoint,
        started_at="2026-04-07T12:00:00",
    )
    for users in [1, 8]:
        reqs = [
            RequestMetrics(
                request_id=i,
                user_id=i,
                task_id=f"P{i + 1}",
                context_profile="medium",
                context_tokens=40000,
                ttft_ms=100 + i * 20,
                total_time_s=2.0,
                completion_tokens=50,
                tok_per_sec=25.0,
                prefill_tok_per_sec=40000.0,
            )
            for i in range(users)
        ]
        run.scenarios.append(
            ScenarioResult(
                num_users=users,
                context_profile="medium",
                context_tokens=40000,
                wall_time_s=2.5,
                requests=reqs,
            )
        )
    return run


# ===========================================================================
# LCP computation
# ===========================================================================


def test_lcp_empty():
    assert compute_lcp([]) == ""


def test_lcp_single():
    assert compute_lcp(["hello world"]) == "hello world"


def test_lcp_identical():
    assert compute_lcp(["abc", "abc", "abc"]) == "abc"


def test_lcp_basic():
    assert compute_lcp(["hello world", "hello there"]) == "hello "


def test_lcp_no_common():
    assert compute_lcp(["abc", "xyz"]) == ""


def test_lcp_growing_conversation():
    r2 = "System prompt. User: hello"
    r3 = "System prompt. User: hello\nAssistant: hi\nUser: more"
    assert compute_lcp([r2, r3]) == r2


# ===========================================================================
# Mask generation
# ===========================================================================


def test_mask_is_deterministic():
    m1 = generate_poison_mask("task-1")
    m2 = generate_poison_mask("task-1")
    assert m1 == m2


def test_mask_is_64_bits():
    m = generate_poison_mask("task-x")
    assert len(m) == 64


def test_mask_is_binary():
    m = generate_poison_mask("task-y")
    assert all(b in (0, 1) for b in m)


def test_different_tasks_different_masks():
    m1 = generate_poison_mask("task-a")
    m2 = generate_poison_mask("task-b")
    assert m1 != m2


def test_mask_roughly_balanced():
    m = generate_poison_mask("balance-test")
    ones = sum(m)
    assert 10 < ones < 54, f"Expected ~32 ones, got {ones}"


# ===========================================================================
# Isolated space finder
# ===========================================================================


def test_find_spaces_basic():
    text = "hello world foo bar"
    positions = find_isolated_spaces(text, 0)
    assert 5 in positions
    assert 11 in positions
    assert 15 in positions
    assert len(positions) == 3


def test_find_spaces_skips_double_spaces():
    text = "hello  world foo"
    positions = find_isolated_spaces(text, 0)
    assert 5 not in positions
    assert 6 not in positions
    assert 12 in positions


def test_find_spaces_after_start():
    text = "hello world foo bar"
    positions = find_isolated_spaces(text, 10)
    assert 5 not in positions
    assert 11 in positions


def test_find_spaces_max_count():
    text = "a b c d e f g h i j k"
    positions = find_isolated_spaces(text, 0, max_count=3)
    assert len(positions) == 3


def test_find_spaces_empty_text():
    assert find_isolated_spaces("", 0) == []


def test_find_spaces_no_spaces():
    assert find_isolated_spaces("nospaces", 0) == []


def test_find_spaces_start_beyond_text():
    assert find_isolated_spaces("hello world", 100) == []


# ===========================================================================
# Scenario LCP
# ===========================================================================


def test_scenario_lcp_shared_system_prompt():
    """Tasks sharing the same system prompt + first user msg have LCP = full R1."""
    prompt = "You are a coding assistant."
    t1 = _make_growing_task("t1", system_prompt=prompt)
    t2 = _make_growing_task("t2", system_prompt=prompt)
    lcp_len = compute_scenario_lcp([t1, t2])
    first_r1 = "".join(m.get("content", "") for m in t1.entries[0].messages)
    assert lcp_len == len(first_r1)


def test_scenario_lcp_different_prompts():
    t1 = _make_growing_task("t1", system_prompt="Alpha system prompt.")
    t2 = _make_growing_task("t2", system_prompt="Alpha different prompt.")
    lcp_len = compute_scenario_lcp([t1, t2])
    assert lcp_len == len("Alpha ")


def test_scenario_lcp_empty_tasks():
    assert compute_scenario_lcp([]) == 0


def test_scenario_lcp_single_task():
    t1 = _make_growing_task("t1")
    lcp_len = compute_scenario_lcp([t1])
    first_text = "".join(m.get("content", "") for m in t1.entries[0].messages)
    assert lcp_len == len(first_text)


# ===========================================================================
# Full task execution poisoning
# ===========================================================================


def test_poison_modifies_requests_after_lcp():
    prompt = "You are a coding assistant."
    task = _make_growing_task("t1", n_requests=5, system_prompt=prompt)
    lcp_len = len(prompt)

    poisoned = poison_task_execution(task, lcp_len, execution_index=0)

    any_modified = False
    for i in range(len(poisoned.entries)):
        orig_text = "".join(m.get("content", "") for m in task.entries[i].messages)
        poisoned_text = "".join(m.get("content", "") for m in poisoned.entries[i].messages)
        if orig_text != poisoned_text:
            any_modified = True

    assert any_modified, "Poisoning should modify at least one request"


def test_poison_preserves_message_count():
    task = _make_growing_task("t1", n_requests=4)
    poisoned = poison_task_execution(task, lcp_len=0, execution_index=0)
    for i in range(len(task.entries)):
        assert len(poisoned.entries[i].messages) == len(task.entries[i].messages)


def test_poison_empty_task_unchanged():
    task = Task(id="empty", entries=[])
    poisoned = poison_task_execution(task, lcp_len=0, execution_index=0)
    assert poisoned.entries == []


def test_poison_deterministic_same_execution():
    task = _make_growing_task("t1", n_requests=4)
    p1 = poison_task_execution(task, lcp_len=0, execution_index=0)
    p2 = poison_task_execution(task, lcp_len=0, execution_index=0)
    for i in range(len(p1.entries)):
        t1 = "".join(m.get("content", "") for m in p1.entries[i].messages)
        t2 = "".join(m.get("content", "") for m in p2.entries[i].messages)
        assert t1 == t2


def test_poison_different_executions_differ():
    """Two repetitions of the same task should get different poisonsets."""
    task = _make_growing_task("t1", n_requests=5)
    p0 = poison_task_execution(task, lcp_len=0, execution_index=0)
    p1 = poison_task_execution(task, lcp_len=0, execution_index=1)

    any_different = False
    for i in range(len(task.entries)):
        text_0 = "".join(m.get("content", "") for m in p0.entries[i].messages)
        text_1 = "".join(m.get("content", "") for m in p1.entries[i].messages)
        if text_0 != text_1:
            any_different = True
            break

    assert any_different, "Different executions should produce different poisoning"


def test_poison_only_doubles_spaces():
    task = _make_growing_task("t1", n_requests=3)
    poisoned = poison_task_execution(task, lcp_len=0, execution_index=0)

    for i in range(len(task.entries)):
        orig = "".join(m.get("content", "") for m in task.entries[i].messages)
        pois = "".join(m.get("content", "") for m in poisoned.entries[i].messages)
        clean = pois.replace("  ", " ")
        while "  " in clean:
            clean = clean.replace("  ", " ")
        assert clean == orig, "Poisoning should only insert extra spaces"


def test_poison_doesnt_exceed_64_modifications():
    entries = [
        RecordingEntry(
            seq=1,
            messages=[{"role": "user", "content": " ".join(["word"] * 200)}],
        ),
    ]
    task = Task(id="many-spaces", entries=entries)
    poisoned = poison_task_execution(task, lcp_len=0, execution_index=0)

    orig = "".join(m.get("content", "") for m in task.entries[0].messages)
    pois = "".join(m.get("content", "") for m in poisoned.entries[0].messages)
    extra_spaces = len(pois) - len(orig)
    assert extra_spaces <= 64


def test_poison_consistent_mask_across_requests():
    """All requests in one execution should use the same mask."""
    task = _make_growing_task("t1", n_requests=5)
    poisoned = poison_task_execution(task, lcp_len=0, execution_index=0)

    for i in range(len(poisoned.entries)):
        orig = "".join(m.get("content", "") for m in task.entries[i].messages)
        pois = "".join(m.get("content", "") for m in poisoned.entries[i].messages)
        if orig == pois:
            continue
        assert len(pois) > len(orig)


def test_poison_lcp_protects_shared_content():
    """Content within the LCP boundary should never be modified."""
    prompt = "You are a helpful coding assistant."
    task = _make_growing_task("t1", n_requests=3, system_prompt=prompt)
    lcp_len = len(prompt)

    poisoned = poison_task_execution(task, lcp_len, execution_index=0)

    for i in range(len(poisoned.entries)):
        orig_text = "".join(m.get("content", "") for m in task.entries[i].messages)
        pois_text = "".join(m.get("content", "") for m in poisoned.entries[i].messages)
        assert pois_text[:lcp_len] == orig_text[:lcp_len]


def test_poison_cross_task_lcp_then_exec():
    """End-to-end: compute LCP across tasks with different R1, then poison."""
    t1 = Task(
        id="t1",
        entries=[
            RecordingEntry(
                seq=1,
                messages=[
                    {"role": "system", "content": "Shared prompt."},
                    {"role": "user", "content": "Task one question alpha"},
                ],
            ),
            RecordingEntry(
                seq=2,
                messages=[
                    {"role": "system", "content": "Shared prompt."},
                    {"role": "user", "content": "Task one question alpha"},
                    {"role": "assistant", "content": "Answer alpha"},
                    {"role": "user", "content": "Follow up one"},
                ],
            ),
        ],
    )
    t2 = Task(
        id="t2",
        entries=[
            RecordingEntry(
                seq=1,
                messages=[
                    {"role": "system", "content": "Shared prompt."},
                    {"role": "user", "content": "Task two question beta"},
                ],
            ),
            RecordingEntry(
                seq=2,
                messages=[
                    {"role": "system", "content": "Shared prompt."},
                    {"role": "user", "content": "Task two question beta"},
                    {"role": "assistant", "content": "Answer beta"},
                    {"role": "user", "content": "Follow up two"},
                ],
            ),
        ],
    )

    lcp_len = compute_scenario_lcp([t1, t2])
    assert lcp_len == len("Shared prompt.Task ")

    p0 = poison_task_execution(t1, lcp_len, execution_index=0)
    p1 = poison_task_execution(t1, lcp_len, execution_index=1)

    for i in range(len(t1.entries)):
        t0_text = "".join(m.get("content", "") for m in p0.entries[i].messages)
        t1_text = "".join(m.get("content", "") for m in p1.entries[i].messages)
        assert t0_text[:lcp_len] == t1_text[:lcp_len]


# ===========================================================================
# Padding poison (per-message space-doubling)
# ===========================================================================


def test_poison_text_spaces_deterministic():
    text = "hello world this is a test"
    a = poison_text_spaces(text, seed="same")
    b = poison_text_spaces(text, seed="same")
    assert a == b


def test_poison_text_spaces_different_seeds():
    text = "hello world this is a test with many spaces in it for poisoning"
    a = poison_text_spaces(text, seed="seed-1")
    b = poison_text_spaces(text, seed="seed-2")
    assert a != b


def test_poison_text_spaces_doubles_spaces():
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    poisoned = poison_text_spaces(text, seed="test")
    assert "  " in poisoned
    assert len(poisoned) > len(text)


def test_poison_messages_returns_new_list():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there friend."},
    ]
    poisoned = poison_messages(msgs, seed="test")
    assert len(poisoned) == 2
    assert poisoned is not msgs
    assert poisoned[0] is not msgs[0]


# ===========================================================================
# Proxy padding with cache defeat
# ===========================================================================


def test_pad_messages_with_cache_defeat():
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Hello"},
    ]
    padded_a = pad_messages_to_target(messages, target_tokens=5000, defeat_cache=True)
    padded_b = pad_messages_to_target(messages, target_tokens=5000, defeat_cache=True)
    assert padded_a[0]["content"] != padded_b[0]["content"]
    assert "[session_id=" not in padded_a[0]["content"]


# ===========================================================================
# Player repetition poisoning invariants
# ===========================================================================


def test_repetitions_produce_distinct_poisoned_payloads():
    """Two repetitions of the same task must yield different serialized bytes.

    This is the invariant that makes `--repetitions N --max-concurrent N`
    a correct replacement for the removed `--users N` flag.
    """
    tasks = _make_poisonable_tasks()
    lcp_len = compute_scenario_lcp(tasks)
    task = tasks[0]

    poisoned_a = poison_task_execution(task, lcp_len, execution_index=0)
    poisoned_b = poison_task_execution(task, lcp_len, execution_index=1)

    text_a = _serialize_messages(poisoned_a.entries[0].messages)
    text_b = _serialize_messages(poisoned_b.entries[0].messages)

    assert text_a != text_b, (
        "Two repetitions must produce different poisoned payloads; "
        "if they match, --repetitions loses its cache-busting property."
    )


def test_repetitions_preserve_shared_lcp():
    """The shared LCP (system prompt) must stay byte-identical across reps.

    Realistic mode's whole point is: shared prefix cached, per-execution tail
    varied. If the LCP diverges, every request becomes a full cold prefill.
    """
    tasks = _make_poisonable_tasks()
    lcp_len = compute_scenario_lcp(tasks)
    task = tasks[0]

    poisoned_a = poison_task_execution(task, lcp_len, execution_index=0)
    poisoned_b = poison_task_execution(task, lcp_len, execution_index=1)

    text_a = _serialize_messages(poisoned_a.entries[0].messages)
    text_b = _serialize_messages(poisoned_b.entries[0].messages)

    assert text_a[:lcp_len] == text_b[:lcp_len]


# ===========================================================================
# Report methodology section
# ===========================================================================


def test_report_includes_cache_poisoning_methodology():
    run = _make_run()
    report = generate_report(run)
    assert "space-doubling" in report
    assert "prefix caching" in report
