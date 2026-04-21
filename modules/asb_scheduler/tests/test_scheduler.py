"""Tests for the asb_scheduler module: random policy, interleaved ordering, and CLI integration."""

from __future__ import annotations

import asyncio
import json

import pytest
from click.testing import CliRunner

from asb_scheduler.interleaved import build_interleaved_order, run_interleaved_work_queue
from asb_scheduler.random_policy import build_random_queue
from agentic_swarm_bench.cli import main
from agentic_swarm_bench.scenarios.registry import Task
from agentic_swarm_bench.scenarios.schedule import (
    Schedule,
    build_execution_queue,
    run_work_queue,
)

RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# build_random_queue / build_execution_queue(policy="random")
# ---------------------------------------------------------------------------


def test_execution_queue_random_contains_all():
    tasks = [Task(id="a"), Task(id="b")]
    queue = build_execution_queue(
        tasks,
        Schedule(repetitions=3, policy="random"),
        seed=42,
    )
    ids = sorted((t.id, ei) for t, ei in queue)
    assert ids == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]


def test_execution_queue_random_is_deterministic():
    tasks = [Task(id="a"), Task(id="b"), Task(id="c")]
    sched = Schedule(repetitions=5, policy="random")
    q1 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched, seed=99)]
    q2 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched, seed=99)]
    assert q1 == q2


# ---------------------------------------------------------------------------
# Schedule.seed fallback and explicit seed override
# ---------------------------------------------------------------------------


def test_schedule_seed_attribute_is_honored():
    """Schedule.seed acts as a fallback when no explicit seed= is passed."""
    tasks = [Task(id="a"), Task(id="b"), Task(id="c")]
    sched_a = Schedule(repetitions=4, policy="random", seed=7)
    sched_b = Schedule(repetitions=4, policy="random", seed=7)
    q1 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched_a)]
    q2 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched_b)]
    assert q1 == q2


def test_schedule_explicit_seed_overrides_attribute():
    """Explicit seed= argument wins over Schedule.seed."""
    tasks = [Task(id="a"), Task(id="b"), Task(id="c")]
    sched = Schedule(repetitions=4, policy="random", seed=1)
    q1 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched, seed=999)]
    q2 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched, seed=999)]
    assert q1 == q2
    sched_default = Schedule(repetitions=4, policy="random", seed=999)
    q3 = [(t.id, ei) for t, ei in build_execution_queue(tasks, sched_default)]
    assert q1 == q3


# ---------------------------------------------------------------------------
# run_work_queue integration with random schedule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_queue_integrates_with_schedule():
    """End-to-end: build_execution_queue -> run_work_queue with J slots."""
    tasks = [Task(id=f"t{i}") for i in range(6)]
    queue = build_execution_queue(
        tasks,
        Schedule(repetitions=8, policy="random", seed=42),
    )
    assert len(queue) == 48

    async def worker(sched_task, slot_id):
        task, exec_idx = sched_task
        return (task.id, exec_idx, slot_id)

    results = await run_work_queue(queue, worker, max_concurrent=4)
    assert len(results) == 48
    for (task, exec_idx), result in zip(queue, results):
        assert result[:2] == (task.id, exec_idx)
        assert 0 <= result[2] < 4


# ---------------------------------------------------------------------------
# build_interleaved_order
# ---------------------------------------------------------------------------


def test_interleaved_order_preserves_within_task_ordering():
    """Entry n must always appear before entry n+1 for any task execution."""
    order = build_interleaved_order([5, 3, 4], seed=42)

    per_task: dict[int, list[int]] = {}
    for task_idx, entry_idx in order:
        per_task.setdefault(task_idx, []).append(entry_idx)

    for task_idx, entries in per_task.items():
        assert entries == sorted(entries), (
            f"Task {task_idx} entries out of order: {entries}"
        )
    assert per_task[0] == [0, 1, 2, 3, 4]
    assert per_task[1] == [0, 1, 2]
    assert per_task[2] == [0, 1, 2, 3]


def test_interleaved_order_contains_all_requests():
    """Every (task_exec, entry) pair must appear exactly once."""
    counts = [3, 2, 4]
    order = build_interleaved_order(counts, seed=7)

    assert len(order) == sum(counts)
    expected = set()
    for i, c in enumerate(counts):
        for e in range(c):
            expected.add((i, e))
    assert set(order) == expected


def test_interleaved_order_is_deterministic_with_seed():
    o1 = build_interleaved_order([5, 3, 4], seed=123)
    o2 = build_interleaved_order([5, 3, 4], seed=123)
    assert o1 == o2


def test_interleaved_order_different_seeds_differ():
    o1 = build_interleaved_order([5, 5, 5], seed=1)
    o2 = build_interleaved_order([5, 5, 5], seed=2)
    assert o1 != o2


def test_interleaved_order_single_task():
    """With one task, output is just sequential entries."""
    order = build_interleaved_order([4], seed=42)
    assert order == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_interleaved_order_empty():
    assert build_interleaved_order([], seed=1) == []


def test_interleaved_order_single_entry_tasks():
    """All single-entry tasks: degenerates to a random shuffle."""
    order = build_interleaved_order([1, 1, 1, 1], seed=99)
    assert sorted(order) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert len(order) == 4


def test_interleaved_order_is_actually_interleaved():
    """With multiple tasks, requests should actually mix (not be sequential)."""
    order = build_interleaved_order([10, 10, 10], seed=42)
    task_ids = [t for t, _e in order]
    consecutive_same = sum(1 for a, b in zip(task_ids, task_ids[1:]) if a == b)
    assert consecutive_same < len(order) - 1


# ---------------------------------------------------------------------------
# Schedule + interleaved_random policy
# ---------------------------------------------------------------------------


def test_schedule_accepts_interleaved_random():
    """interleaved_random is a valid policy."""
    sched = Schedule(policy="interleaved_random")
    assert sched.policy == "interleaved_random"


def test_execution_queue_interleaved_random():
    """build_execution_queue with interleaved_random produces all tasks."""
    tasks = [Task(id="a"), Task(id="b")]
    queue = build_execution_queue(
        tasks, Schedule(repetitions=2, policy="interleaved_random")
    )
    ids = sorted((t.id, ei) for t, ei in queue)
    assert ids == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]


# ---------------------------------------------------------------------------
# run_interleaved_work_queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interleaved_work_queue_empty():
    async def worker(task_idx, entry_idx, slot_id):
        return (task_idx, entry_idx)

    results = await run_interleaved_work_queue([], worker, max_concurrent=4)
    assert results == []


@pytest.mark.asyncio
async def test_interleaved_work_queue_respects_ordering():
    """Within-task entries must execute in order even with J>1."""
    execution_log: list[tuple[int, int]] = []
    lock = asyncio.Lock()

    async def worker(task_idx, entry_idx, slot_id):
        async with lock:
            execution_log.append((task_idx, entry_idx))
        await asyncio.sleep(0.001)
        return (task_idx, entry_idx)

    order = build_interleaved_order([3, 3, 3], seed=42)
    await run_interleaved_work_queue(order, worker, max_concurrent=3)

    per_task: dict[int, list[int]] = {}
    for task_idx, entry_idx in execution_log:
        per_task.setdefault(task_idx, []).append(entry_idx)

    for task_idx, entries in per_task.items():
        assert entries == sorted(entries), (
            f"Task {task_idx} executed out of order: {entries}"
        )


@pytest.mark.asyncio
async def test_interleaved_work_queue_all_results_collected():
    async def worker(task_idx, entry_idx, slot_id):
        return task_idx * 100 + entry_idx

    order = build_interleaved_order([3, 2, 4], seed=7)
    results = await run_interleaved_work_queue(order, worker, max_concurrent=2)

    assert len(results) == 9
    result_set = {r for r in results if r is not None}
    expected = set()
    for task_idx, count in enumerate([3, 2, 4]):
        for entry_idx in range(count):
            expected.add(task_idx * 100 + entry_idx)
    assert result_set == expected


@pytest.mark.asyncio
async def test_interleaved_work_queue_concurrent():
    """J>1 should actually run requests from different tasks in parallel."""
    live = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(task_idx, entry_idx, slot_id):
        nonlocal live, peak
        async with lock:
            live += 1
            peak = max(peak, live)
        await asyncio.sleep(0.02)
        async with lock:
            live -= 1
        return (task_idx, entry_idx)

    order = build_interleaved_order([1, 1, 1, 1, 1], seed=42)
    await run_interleaved_work_queue(order, worker, max_concurrent=3)
    assert peak >= 2, f"Expected parallel execution, but peak concurrency was {peak}"


@pytest.mark.asyncio
async def test_interleaved_work_queue_j1_sequential():
    """With J=1, all requests execute sequentially in order."""
    execution_log: list[tuple[int, int]] = []

    async def worker(task_idx, entry_idx, slot_id):
        execution_log.append((task_idx, entry_idx))
        assert slot_id == 0
        return (task_idx, entry_idx)

    order = build_interleaved_order([2, 2], seed=42)
    await run_interleaved_work_queue(order, worker, max_concurrent=1)
    assert execution_log == order


# ---------------------------------------------------------------------------
# CLI integration: agent default policy, interleaved_random in help, seed echo
# ---------------------------------------------------------------------------


def test_agent_default_policy_is_random():
    """Agent mode must default to random policy so cache free-rides don't happen silently."""
    result = RUNNER.invoke(main, ["agent", "--help"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "[default: random]" in flat


def test_interleaved_random_policy_in_help():
    """Both agent and replay must accept interleaved_random as a policy."""
    for cmd in ("agent", "replay"):
        result = RUNNER.invoke(main, [cmd, "--help"])
        assert result.exit_code == 0
        assert "interleaved_random" in result.output, (
            f"interleaved_random missing from {cmd} --help"
        )


def test_replay_seed_shown_in_dry_run_output(tmp_path):
    """When --seed is passed, the dry-run summary must echo it so you can reproduce."""
    scenario = tmp_path / "scenario.jsonl"
    scenario.write_text(
        json.dumps({
            "seq": 1,
            "experiment_id": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "max_tokens": 100,
            "stream": True,
        })
        + "\n"
    )

    result = RUNNER.invoke(
        main,
        [
            "replay",
            "--endpoint", "http://localhost:8000",
            "--model", "test-model",
            "--scenario", str(scenario),
            "--repetitions", "3",
            "--policy", "random",
            "--seed", "42",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"dry-run failed:\n{result.output}"
    assert "seed=42" in result.output
