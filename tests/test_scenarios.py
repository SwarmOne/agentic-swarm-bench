"""Tests for scenario registry and schedule (public functionality)."""

import json

import pytest

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
    assert task_ids == {
        "build-rest-api", "csv-parser-cli", "websocket-chat",
        "markdown-renderer", "state-machine",
    }
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
    s = load_scenario(path)
    assert len(s.tasks) == 1
    assert s.tasks[0].total_requests == 1


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


# ---------------------------------------------------------------------------
# min_lcp_length
# ---------------------------------------------------------------------------


def test_min_lcp_length_parsed_from_manifest(tmp_path):
    scenario_dir = tmp_path / "lcp-test"
    scenario_dir.mkdir()

    rec = scenario_dir / "task.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    manifest = {
        "name": "lcp-test",
        "min_lcp_length": 5000,
        "tasks": [{"id": "t", "recording": "task.jsonl"}],
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    scenario = load_scenario(scenario_dir)
    assert scenario.min_lcp_length == 5000


def test_min_lcp_length_none_when_absent(tmp_path):
    path = _make_scenario_json(tmp_path)
    scenario = load_scenario(path)
    assert scenario.min_lcp_length is None


def test_min_lcp_length_parsed_from_standalone_json(tmp_path):
    rec = tmp_path / "t.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    manifest = {
        "name": "standalone-lcp",
        "min_lcp_length": 1234,
        "tasks": [{"id": "t", "recording": "t.jsonl"}],
    }
    path = tmp_path / "standalone.json"
    path.write_text(json.dumps(manifest))

    scenario = load_scenario(path)
    assert scenario.min_lcp_length == 1234


def test_min_lcp_length_preserved_through_task_filter(tmp_path):
    path = _make_scenario_json(tmp_path)
    with open(path) as f:
        manifest = json.load(f)
    manifest["min_lcp_length"] = 9999
    with open(path, "w") as f:
        json.dump(manifest, f)

    scenario = get_scenario(str(path), task_filter="session-a")
    assert scenario.min_lcp_length == 9999
    assert len(scenario.tasks) == 1


# ---------------------------------------------------------------------------
# Built-in filename resolution (-s name/manifest)
# ---------------------------------------------------------------------------


def test_builtin_json_file_resolution(tmp_path, monkeypatch):
    """Resolving 'mydir/small' finds data/mydir/small.json."""
    import agentic_swarm_bench.scenarios.registry as reg

    data_dir = tmp_path / "data"
    scenario_dir = data_dir / "mydir"
    scenario_dir.mkdir(parents=True)

    rec = scenario_dir / "t.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    manifest = {"name": "small-test", "tasks": [{"id": "t", "recording": "t.jsonl"}]}
    (scenario_dir / "small.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(reg, "BUILTIN_DIR", data_dir)
    scenario = get_scenario("mydir/small")
    assert scenario.name == "small-test"


def test_builtin_json_file_explicit_extension(tmp_path, monkeypatch):
    """Resolving 'mydir/small.json' also works (literal file fallback)."""
    import agentic_swarm_bench.scenarios.registry as reg

    data_dir = tmp_path / "data"
    scenario_dir = data_dir / "mydir"
    scenario_dir.mkdir(parents=True)

    rec = scenario_dir / "t.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    manifest = {"name": "small-explicit", "tasks": [{"id": "t", "recording": "t.jsonl"}]}
    (scenario_dir / "small.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(reg, "BUILTIN_DIR", data_dir)
    scenario = get_scenario("mydir/small.json")
    assert scenario.name == "small-explicit"


def test_builtin_dir_still_takes_priority(tmp_path, monkeypatch):
    """A directory with scenario.json still wins over name.json resolution."""
    import agentic_swarm_bench.scenarios.registry as reg

    data_dir = tmp_path / "data"
    scenario_dir = data_dir / "mydir"
    scenario_dir.mkdir(parents=True)

    rec = scenario_dir / "t.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    manifest = {"name": "from-dir", "tasks": [{"id": "t", "recording": "t.jsonl"}]}
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(reg, "BUILTIN_DIR", data_dir)
    scenario = get_scenario("mydir")
    assert scenario.name == "from-dir"


def test_list_builtin_discovers_extra_json_files(tmp_path, monkeypatch):
    """list_builtin_scenarios finds both scenario.json and sibling .json manifests."""
    import agentic_swarm_bench.scenarios.registry as reg

    data_dir = tmp_path / "data"
    scenario_dir = data_dir / "mydir"
    scenario_dir.mkdir(parents=True)

    rec = scenario_dir / "t.jsonl"
    rec.write_text(
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )

    for fname, sname in [("scenario.json", "full"), ("small.json", "small")]:
        manifest = {"name": sname, "tasks": [{"id": "t", "recording": "t.jsonl"}]}
        (scenario_dir / fname).write_text(json.dumps(manifest))

    monkeypatch.setattr(reg, "BUILTIN_DIR", data_dir)
    scenarios = list_builtin_scenarios()
    names = {s["name"] for s in scenarios}
    assert "full" in names
    assert "small" in names


# ---------------------------------------------------------------------------
# Null-safety: "messages": null and "content": null in JSONL
# ---------------------------------------------------------------------------


def test_parse_entry_messages_null_becomes_empty_list():
    """When a JSONL line has "messages": null, the entry should get [] not None."""
    from agentic_swarm_bench.scenarios.registry import _parse_entry

    entry = _parse_entry({"seq": 1, "messages": None})
    assert entry.messages == []
    assert isinstance(entry.messages, list)


def test_parse_entry_messages_absent_becomes_empty_list():
    from agentic_swarm_bench.scenarios.registry import _parse_entry

    entry = _parse_entry({"seq": 1})
    assert entry.messages == []


def test_total_tokens_approx_with_null_messages():
    """Task.total_tokens_approx must not crash when an entry has messages=[]."""
    entry = RecordingEntry(seq=1, messages=[])
    task = Task(id="t", entries=[entry])
    assert task.total_tokens_approx == 0


def test_load_jsonl_null_messages(tmp_path):
    """A JSONL file with null messages loads without error."""
    from agentic_swarm_bench.scenarios.registry import _load_jsonl

    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"seq": 1, "messages": null}\n')
    entries = _load_jsonl(jsonl)
    assert len(entries) == 1
    assert entries[0].messages == []


def test_content_null_in_message_does_not_crash():
    """len(m.get("content") or "") handles null content without TypeError."""
    msgs = [{"role": "user", "content": None}]
    tok = sum(len(m.get("content") or "") for m in msgs) // 4
    assert tok == 0


def test_total_tokens_approx_with_content_null():
    """Task.total_tokens_approx handles content: null gracefully."""
    entry = RecordingEntry(
        seq=1,
        messages=[{"role": "user", "content": None}],
    )
    task = Task(id="t", entries=[entry])
    assert task.total_tokens_approx == 0
