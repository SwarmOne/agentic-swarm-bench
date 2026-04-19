"""Tests for scenario registry, schedule, poisoning, and replay."""

import json

import pytest

from agentic_swarm_bench.scenarios.poison import (
    compute_lcp,
    compute_scenario_lcp,
    find_isolated_spaces,
    generate_poison_mask,
    poison_task_execution,
)
from agentic_swarm_bench.scenarios.registry import (
    RecordingEntry,
    Scenario,
    Task,
    get_scenario,
    list_builtin_scenarios,
    load_scenario,
)
from agentic_swarm_bench.scenarios.schedule import (
    Schedule,
    build_execution_queue,
    run_work_queue,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_jsonl(tmp_path, n_entries=3):
    """Create a sample JSONL scenario file."""
    path = tmp_path / "test_scenario.jsonl"
    entries = []
    for i in range(n_entries):
        entry = {
            "seq": i + 1,
            "experiment_id": "test123",
            "timestamp": f"2026-04-11T12:00:0{i}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Fix the bug in function_{i}"},
            ],
            "model": "test-model",
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True,
            "ttft_ms": 100.0 + i * 50,
            "total_time_s": 2.0,
            "completion_tokens": 50,
            "tok_per_sec": 25.0,
        }
        entries.append(entry)

    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return path


def _make_scenario_dir(tmp_path, task_count=2, entries_per_task=3):
    """Create a scenario directory with manifest and task recordings."""
    scenario_dir = tmp_path / "test-scenario"
    scenario_dir.mkdir()

    tasks = []
    for t in range(task_count):
        task_id = f"task-{t}"
        recording_file = f"{task_id}.jsonl"

        entries = []
        for i in range(entries_per_task):
            entries.append(
                {
                    "seq": i + 1,
                    "experiment_id": f"exp-{t}",
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": f"Task {t} request {i}"},
                    ],
                    "model": "test-model",
                    "max_tokens": 512,
                }
            )

        with open(scenario_dir / recording_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        tasks.append({"id": task_id, "name": f"Task {t}", "recording": recording_file})

    manifest = {
        "name": "test-scenario",
        "description": "A test scenario",
        "tasks": tasks,
    }
    with open(scenario_dir / "scenario.json", "w") as f:
        json.dump(manifest, f)

    return scenario_dir


# ---------------------------------------------------------------------------
# Registry: JSONL loading (backward compat)
# ---------------------------------------------------------------------------


def test_load_scenario_from_jsonl(tmp_path):
    path = _make_scenario_jsonl(tmp_path)
    s = load_scenario(path)
    assert s.total_requests == 3
    assert s.name == "test_scenario"
    assert len(s.tasks) == 1
    assert s.tasks[0].id == "default"
    assert len(s.experiment_ids) == 1
    assert "test123" in s.experiment_ids


def test_scenario_entry_fields(tmp_path):
    path = _make_scenario_jsonl(tmp_path, n_entries=1)
    s = load_scenario(path)
    entry = s.tasks[0].entries[0]
    assert entry.seq == 1
    assert entry.model == "test-model"
    assert entry.max_tokens == 512
    assert entry.stream is True
    assert entry.ttft_ms == 100.0
    assert len(entry.messages) == 2


def test_scenario_summary(tmp_path):
    path = _make_scenario_jsonl(tmp_path)
    s = load_scenario(path)
    summary = s.summary()
    assert summary["name"] == "test_scenario"
    assert summary["requests"] == 3
    assert summary["experiments"] == 1
    assert summary["approx_tokens"] > 0


def test_scenario_total_tokens(tmp_path):
    path = _make_scenario_jsonl(tmp_path)
    s = load_scenario(path)
    assert s.total_tokens_approx > 0


def test_get_scenario_by_path(tmp_path):
    path = _make_scenario_jsonl(tmp_path)
    s = get_scenario(str(path))
    assert s.total_requests == 3


def test_get_scenario_not_found():
    with pytest.raises(FileNotFoundError):
        get_scenario("nonexistent_scenario_xyz")


def test_list_builtin_scenarios():
    scenarios = list_builtin_scenarios()
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 2
    names = {s["name"] for s in scenarios}
    assert "js-coding-opus" in names
    assert "trivial-qa" in names


def test_builtin_js_coding_opus():
    s = get_scenario("js-coding-opus")
    assert s.name == "js-coding-opus"
    assert s.model == "claude-opus-4-6"
    assert len(s.tasks) == 5
    task_ids = {t.id for t in s.tasks}
    assert task_ids == {"build-rest-api", "csv-parser-cli", "websocket-chat", "markdown-renderer", "state-machine"}
    for task in s.tasks:
        assert task.total_requests >= 2
        assert len(task.entries[0].messages) >= 2


def test_builtin_trivial_qa():
    s = get_scenario("trivial-qa")
    assert s.name == "trivial-qa"
    assert s.model == "claude-opus-4-6"
    assert len(s.tasks) == 5
    for task in s.tasks:
        assert task.total_requests == 1
        assert len(task.entries[0].messages) == 2
        assert task.entries[0].messages[0]["role"] == "system"
        assert task.entries[0].messages[1]["role"] == "user"


def test_builtin_js_coding_opus_task_filter():
    s = get_scenario("js-coding-opus", task_filter="build-rest-api")
    assert len(s.tasks) == 1
    assert s.tasks[0].id == "build-rest-api"


def test_empty_scenario_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    s = load_scenario(path)
    assert s.total_requests == 0
    assert s.tasks[0].entries == []


def test_recording_entry_defaults():
    entry = RecordingEntry()
    assert entry.seq == 0
    assert entry.messages == []
    assert entry.stream is True
    assert entry.ttft_ms is None


def test_scenario_multiple_experiments(tmp_path):
    path = tmp_path / "multi.jsonl"
    entries = [
        {"seq": 1, "experiment_id": "exp_a", "messages": [{"role": "user", "content": "a"}]},
        {"seq": 2, "experiment_id": "exp_b", "messages": [{"role": "user", "content": "b"}]},
        {"seq": 3, "experiment_id": "exp_a", "messages": [{"role": "user", "content": "c"}]},
    ]
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    s = load_scenario(path)
    assert s.total_requests == 3
    assert len(s.experiment_ids) == 2
    assert set(s.experiment_ids) == {"exp_a", "exp_b"}


def test_load_scenario_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path / "does_not_exist.jsonl")


def test_scenario_with_blank_lines(tmp_path):
    path = tmp_path / "blanks.jsonl"
    content = (
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]})
        + "\n\n"
        + json.dumps({"seq": 2, "messages": [{"role": "user", "content": "bye"}]})
        + "\n\n\n"
    )
    path.write_text(content)
    s = load_scenario(path)
    assert s.total_requests == 2


def test_scenario_with_missing_fields(tmp_path):
    path = tmp_path / "minimal.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n")
    s = load_scenario(path)
    assert s.total_requests == 1
    entry = s.tasks[0].entries[0]
    assert entry.seq == 0
    assert entry.experiment_id == ""
    assert entry.model == ""
    assert entry.max_tokens == 4096
    assert entry.temperature == 1.0
    assert entry.ttft_ms is None
    assert entry.tok_per_sec is None


def test_scenario_with_empty_messages(tmp_path):
    path = tmp_path / "no_msgs.jsonl"
    path.write_text(json.dumps({"seq": 1, "messages": []}) + "\n")
    s = load_scenario(path)
    assert s.total_requests == 1
    assert s.total_tokens_approx == 0


def test_scenario_messages_with_no_content_key(tmp_path):
    path = tmp_path / "no_content.jsonl"
    entry = {
        "seq": 1,
        "messages": [
            {"role": "user"},
            {"role": "assistant", "tool_calls": [{"type": "function"}]},
        ],
    }
    path.write_text(json.dumps(entry) + "\n")
    s = load_scenario(path)
    assert s.total_tokens_approx == 0


def test_scenario_with_malformed_json_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"seq": 1}\n{bad json}\n')
    with pytest.raises(json.JSONDecodeError):
        load_scenario(path)


def test_scenario_path_preserved(tmp_path):
    path = _make_scenario_jsonl(tmp_path)
    s = load_scenario(path)
    assert s.path == str(path)


def test_scenario_name_from_stem(tmp_path):
    path = tmp_path / "my-session-recording.jsonl"
    path.write_text(json.dumps({"seq": 1, "messages": []}) + "\n")
    s = load_scenario(path)
    assert s.name == "my-session-recording"


def test_scenario_empty_experiment_ids_excluded():
    task = Task(
        entries=[
            RecordingEntry(seq=1, experiment_id=""),
            RecordingEntry(seq=2, experiment_id="exp1"),
        ]
    )
    scenario = Scenario(tasks=[task])
    assert scenario.experiment_ids == ["exp1"]


def test_get_scenario_by_name_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        get_scenario("definitely_not_a_real_scenario")


# ---------------------------------------------------------------------------
# Registry: Directory loading
# ---------------------------------------------------------------------------


def test_load_scenario_from_directory(tmp_path):
    scenario_dir = _make_scenario_dir(tmp_path)
    s = load_scenario(scenario_dir)
    assert s.name == "test-scenario"
    assert s.description == "A test scenario"
    assert len(s.tasks) == 2
    assert s.total_requests == 6  # 2 tasks * 3 entries each
    assert s.tasks[0].id == "task-0"
    assert s.tasks[1].id == "task-1"


def test_load_scenario_dir_missing_manifest(tmp_path):
    empty_dir = tmp_path / "no-manifest"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="scenario.json"):
        load_scenario(empty_dir)


def test_load_scenario_dir_missing_recording(tmp_path):
    scenario_dir = tmp_path / "bad-ref"
    scenario_dir.mkdir()
    manifest = {"tasks": [{"id": "t1", "recording": "nonexistent.jsonl"}]}
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="nonexistent.jsonl"):
        load_scenario(scenario_dir)


def test_scenario_all_entries_flattens(tmp_path):
    scenario_dir = _make_scenario_dir(tmp_path, task_count=3, entries_per_task=2)
    s = load_scenario(scenario_dir)
    flat = s.all_entries
    assert len(flat) == 6
    assert all(isinstance(e, RecordingEntry) for e in flat)


def test_get_scenario_by_dir_path(tmp_path):
    scenario_dir = _make_scenario_dir(tmp_path)
    s = get_scenario(str(scenario_dir))
    assert s.name == "test-scenario"


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def test_schedule_defaults():
    s = Schedule()
    assert s.repetitions == 1
    assert s.max_concurrent == 10
    assert s.policy == "round_robin"


def test_schedule_validation_repetitions():
    with pytest.raises(ValueError, match="repetitions"):
        Schedule(repetitions=0)


def test_schedule_validation_max_concurrent():
    with pytest.raises(ValueError, match="max_concurrent"):
        Schedule(max_concurrent=0)


def test_schedule_validation_policy():
    with pytest.raises(ValueError, match="policy"):
        Schedule(policy="invalid")


def test_execution_queue_round_robin():
    tasks = [Task(id="a"), Task(id="b"), Task(id="c")]
    queue = build_execution_queue(tasks, Schedule(repetitions=2, policy="round_robin"))
    ids = [(t.id, ei) for t, ei in queue]
    assert ids == [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1)]


def test_execution_queue_sequential():
    tasks = [Task(id="a"), Task(id="b")]
    queue = build_execution_queue(tasks, Schedule(repetitions=3, policy="sequential"))
    ids = [(t.id, ei) for t, ei in queue]
    assert ids == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]


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


def test_execution_queue_empty_tasks():
    queue = build_execution_queue([], Schedule(repetitions=5))
    assert queue == []


def test_execution_queue_single_task():
    tasks = [Task(id="solo")]
    queue = build_execution_queue(tasks, Schedule(repetitions=3, policy="round_robin"))
    assert [(t.id, ei) for t, ei in queue] == [("solo", 0), ("solo", 1), ("solo", 2)]


def test_execution_queue_tracks_exec_index():
    """Each repetition of a task gets a unique execution index."""
    tasks = [Task(id="x"), Task(id="y")]
    queue = build_execution_queue(tasks, Schedule(repetitions=4, policy="sequential"))
    x_indices = [ei for t, ei in queue if t.id == "x"]
    y_indices = [ei for t, ei in queue if t.id == "y"]
    assert x_indices == [0, 1, 2, 3]
    assert y_indices == [0, 1, 2, 3]


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
# run_work_queue: pool of J workers pulling from head of pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_queue_empty_returns_empty():
    results = await run_work_queue([], lambda item, slot: None, max_concurrent=4)
    assert results == []


@pytest.mark.asyncio
async def test_work_queue_preserves_input_order_in_results():
    """results[i] must correspond to worker(queue[i], ...), regardless of slot."""
    import asyncio

    async def worker(item, slot_id):
        await asyncio.sleep(0.001 * (10 - item))
        return item * 2

    out = await run_work_queue(list(range(10)), worker, max_concurrent=3)
    assert out == [i * 2 for i in range(10)]


@pytest.mark.asyncio
async def test_work_queue_respects_max_concurrent():
    """At most J workers run simultaneously."""
    import asyncio

    live = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(item, slot_id):
        nonlocal live, peak
        async with lock:
            live += 1
            peak = max(peak, live)
        await asyncio.sleep(0.01)
        async with lock:
            live -= 1
        return slot_id

    await run_work_queue(list(range(20)), worker, max_concurrent=4)
    assert peak <= 4
    assert peak >= 1


@pytest.mark.asyncio
async def test_work_queue_slot_ids_bounded_by_j():
    """slot_id passed to worker is always in [0, J)."""

    async def worker(item, slot_id):
        return slot_id

    slot_ids = await run_work_queue(list(range(30)), worker, max_concurrent=3)
    assert set(slot_ids).issubset({0, 1, 2})


@pytest.mark.asyncio
async def test_work_queue_j_greater_than_queue_does_not_hang():
    """If J > len(queue), excess workers must exit instead of hanging."""
    import asyncio

    async def worker(item, slot_id):
        return item

    out = await asyncio.wait_for(
        run_work_queue([1, 2], worker, max_concurrent=10),
        timeout=1.0,
    )
    assert out == [1, 2]


@pytest.mark.asyncio
async def test_work_queue_drains_all_items_even_if_one_is_slow():
    """A slow item on one slot must not prevent other slots from draining the rest."""
    import asyncio

    async def worker(item, slot_id):
        if item == 0:
            await asyncio.sleep(0.2)
        return item

    out = await asyncio.wait_for(
        run_work_queue(list(range(8)), worker, max_concurrent=4),
        timeout=1.5,
    )
    assert out == list(range(8))


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
# Poison: LCP computation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Poison: mask generation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Poison: isolated space finder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Poison: scenario LCP
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


# ---------------------------------------------------------------------------
# Poison: full task execution poisoning
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Player helpers (from old test_workloads.py)
# ---------------------------------------------------------------------------


def test_bucket_label_thresholds():
    from agentic_swarm_bench.scenarios.player import _bucket_label

    assert _bucket_label(0) == "fresh"
    assert _bucket_label(5_000) == "fresh"
    assert _bucket_label(9_999) == "fresh"
    assert _bucket_label(10_000) == "short"
    assert _bucket_label(29_999) == "short"
    assert _bucket_label(30_000) == "medium"
    assert _bucket_label(54_999) == "medium"
    assert _bucket_label(55_000) == "long"
    assert _bucket_label(84_999) == "long"
    assert _bucket_label(85_000) == "full"
    assert _bucket_label(149_999) == "full"
    assert _bucket_label(150_000) == "xl"
    assert _bucket_label(299_999) == "xl"
    assert _bucket_label(300_000) == "xxl"
    assert _bucket_label(1_000_000) == "xxl"


# ---------------------------------------------------------------------------
# Standalone scenario JSON loading
# ---------------------------------------------------------------------------


def _make_scenario_json(tmp_path, name="test-scenario", tasks=None):
    """Create a scenario JSON + recording JSONL files, return scenario JSON path."""
    if tasks is None:
        tasks = [
            {"id": "session-a", "name": "Session A", "entries": 3},
            {"id": "session-b", "name": "Session B", "entries": 2},
        ]

    task_defs = []
    for task in tasks:
        rec_file = f"{task['id']}.jsonl"
        rec_path = tmp_path / rec_file
        with open(rec_path, "w") as f:
            for i in range(task["entries"]):
                entry = {
                    "seq": i + 1,
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": f"Request {i} for {task['id']}"},
                    ],
                    "model": "test-model",
                    "max_tokens": 512,
                }
                f.write(json.dumps(entry) + "\n")
        task_defs.append({"id": task["id"], "name": task["name"], "recording": rec_file})

    scenario_path = tmp_path / "scenario.json"
    manifest = {
        "name": name,
        "description": f"Test scenario: {name}",
        "model": "test-model",
        "tasks": task_defs,
    }
    with open(scenario_path, "w") as f:
        json.dump(manifest, f)

    return scenario_path


def test_load_scenario_json_basic(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = load_scenario(path)
    assert scenario.name == "test-scenario"
    assert scenario.description == "Test scenario: test-scenario"
    assert scenario.model == "test-model"
    assert len(scenario.tasks) == 2
    assert scenario.tasks[0].name == "Session A"
    assert scenario.tasks[1].name == "Session B"
    assert scenario.total_requests == 5  # 3 + 2


def test_load_scenario_json_single_task(tmp_path):
    path = _make_scenario_json(
        tmp_path,
        name="single",
        tasks=[{"id": "only-task", "name": "Only Task", "entries": 4}],
    )
    scenario = load_scenario(path)
    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].id == "only-task"
    assert scenario.total_requests == 4


def test_load_scenario_json_missing_file():
    with pytest.raises(FileNotFoundError, match="Scenario not found"):
        load_scenario("/does/not/exist/scenario.json")


def test_load_scenario_json_missing_recording(tmp_path):
    scenario_path = tmp_path / "bad.json"
    manifest = {
        "name": "bad",
        "tasks": [{"id": "t1", "recording": "nonexistent.jsonl"}],
    }
    with open(scenario_path, "w") as f:
        json.dump(manifest, f)
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        load_scenario(scenario_path)


def test_load_scenario_json_relative_paths(tmp_path):
    sub = tmp_path / "recordings"
    sub.mkdir()
    rec_path = sub / "deep.jsonl"
    rec_path.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )

    scenario_path = tmp_path / "scenario.json"
    manifest = {
        "name": "relative-test",
        "tasks": [{"id": "deep", "name": "Deep", "recording": "recordings/deep.jsonl"}],
    }
    with open(scenario_path, "w") as f:
        json.dump(manifest, f)

    scenario = load_scenario(scenario_path)
    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].name == "Deep"
    assert scenario.total_requests == 1


def test_load_scenario_json_with_explicit_ids(tmp_path):
    rec_path = tmp_path / "r.jsonl"
    rec_path.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )

    scenario_path = tmp_path / "scenario.json"
    manifest = {
        "name": "id-test",
        "tasks": [{"id": "custom-id", "name": "Friendly Name", "recording": "r.jsonl"}],
    }
    with open(scenario_path, "w") as f:
        json.dump(manifest, f)

    scenario = load_scenario(scenario_path)
    assert scenario.tasks[0].id == "custom-id"
    assert scenario.tasks[0].name == "Friendly Name"


def test_load_scenario_json_empty_tasks(tmp_path):
    scenario_path = tmp_path / "empty.json"
    manifest = {"name": "empty", "tasks": []}
    with open(scenario_path, "w") as f:
        json.dump(manifest, f)

    scenario = load_scenario(scenario_path)
    assert len(scenario.tasks) == 0
    assert scenario.total_requests == 0


def test_load_scenario_json_preserves_path(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = load_scenario(path)
    assert scenario.path == str(path)


def test_load_scenario_json_preserves_model(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = load_scenario(path)
    assert scenario.model == "test-model"
    summary = scenario.summary()
    assert summary["model"] == "test-model"


# ---------------------------------------------------------------------------
# Task filter (--task flag)
# ---------------------------------------------------------------------------


def test_get_scenario_with_task_filter(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = get_scenario(str(path), task_filter="session-a")
    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].id == "session-a"
    assert scenario.total_requests == 3


def test_get_scenario_task_filter_not_found(tmp_path):
    path = _make_scenario_json(tmp_path)
    with pytest.raises(FileNotFoundError, match="Task 'nonexistent'"):
        get_scenario(str(path), task_filter="nonexistent")


def test_get_scenario_task_filter_from_directory(tmp_path):
    scenario_dir = _make_scenario_dir(tmp_path)
    scenario = get_scenario(str(scenario_dir), task_filter="task-0")
    assert len(scenario.tasks) == 1
    assert scenario.tasks[0].id == "task-0"


def test_get_scenario_no_filter_returns_all(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = get_scenario(str(path))
    assert len(scenario.tasks) == 2
