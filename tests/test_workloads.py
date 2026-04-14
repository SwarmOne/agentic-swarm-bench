"""Tests for workload recording, registry, and replay."""

import json

import pytest

from agentic_swarm_bench.workloads.registry import (
    Workload,
    WorkloadEntry,
    get_workload,
    list_builtin_workloads,
    load_workload,
)


def _make_workload_file(tmp_path, n_entries=3):
    """Create a sample JSONL workload file."""
    path = tmp_path / "test_workload.jsonl"
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


def test_load_workload(tmp_path):
    path = _make_workload_file(tmp_path)
    wl = load_workload(path)
    assert wl.total_requests == 3
    assert wl.name == "test_workload"
    assert len(wl.experiment_ids) == 1
    assert "test123" in wl.experiment_ids


def test_workload_entry_fields(tmp_path):
    path = _make_workload_file(tmp_path, n_entries=1)
    wl = load_workload(path)
    entry = wl.entries[0]
    assert entry.seq == 1
    assert entry.model == "test-model"
    assert entry.max_tokens == 512
    assert entry.stream is True
    assert entry.ttft_ms == 100.0
    assert len(entry.messages) == 2


def test_workload_summary(tmp_path):
    path = _make_workload_file(tmp_path)
    wl = load_workload(path)
    summary = wl.summary()
    assert summary["name"] == "test_workload"
    assert summary["requests"] == 3
    assert summary["experiments"] == 1
    assert summary["approx_tokens"] > 0


def test_workload_total_tokens(tmp_path):
    path = _make_workload_file(tmp_path)
    wl = load_workload(path)
    assert wl.total_tokens_approx > 0


def test_get_workload_by_path(tmp_path):
    path = _make_workload_file(tmp_path)
    wl = get_workload(str(path))
    assert wl.total_requests == 3


def test_get_workload_not_found():
    with pytest.raises(FileNotFoundError):
        get_workload("nonexistent_workload_xyz")


def test_list_builtin_workloads():
    workloads = list_builtin_workloads()
    assert isinstance(workloads, list)


def test_empty_workload_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    wl = load_workload(path)
    assert wl.total_requests == 0
    assert wl.entries == []


def test_workload_entry_defaults():
    entry = WorkloadEntry()
    assert entry.seq == 0
    assert entry.messages == []
    assert entry.stream is True
    assert entry.ttft_ms is None


def test_workload_multiple_experiments(tmp_path):
    path = tmp_path / "multi.jsonl"
    entries = [
        {"seq": 1, "experiment_id": "exp_a", "messages": [{"role": "user", "content": "a"}]},
        {"seq": 2, "experiment_id": "exp_b", "messages": [{"role": "user", "content": "b"}]},
        {"seq": 3, "experiment_id": "exp_a", "messages": [{"role": "user", "content": "c"}]},
    ]
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    wl = load_workload(path)
    assert wl.total_requests == 3
    assert len(wl.experiment_ids) == 2
    assert set(wl.experiment_ids) == {"exp_a", "exp_b"}


# --- Edge cases ---


def test_load_workload_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workload(tmp_path / "does_not_exist.jsonl")


def test_workload_with_blank_lines(tmp_path):
    """JSONL files with blank lines should be handled gracefully."""
    path = tmp_path / "blanks.jsonl"
    content = (
        json.dumps({"seq": 1, "messages": [{"role": "user", "content": "hi"}]})
        + "\n\n"
        + json.dumps({"seq": 2, "messages": [{"role": "user", "content": "bye"}]})
        + "\n\n\n"
    )
    path.write_text(content)
    wl = load_workload(path)
    assert wl.total_requests == 2


def test_workload_with_missing_fields(tmp_path):
    """Entries missing optional fields should use defaults."""
    path = tmp_path / "minimal.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n")
    wl = load_workload(path)
    assert wl.total_requests == 1
    entry = wl.entries[0]
    assert entry.seq == 0
    assert entry.experiment_id == ""
    assert entry.model == ""
    assert entry.max_tokens == 4096
    assert entry.temperature == 1.0
    assert entry.ttft_ms is None
    assert entry.tok_per_sec is None


def test_workload_with_empty_messages(tmp_path):
    """Entry with empty messages list."""
    path = tmp_path / "no_msgs.jsonl"
    path.write_text(json.dumps({"seq": 1, "messages": []}) + "\n")
    wl = load_workload(path)
    assert wl.total_requests == 1
    assert wl.total_tokens_approx == 0


def test_workload_messages_with_no_content_key(tmp_path):
    """Messages missing the 'content' key should not crash token counting."""
    path = tmp_path / "no_content.jsonl"
    entry = {
        "seq": 1,
        "messages": [
            {"role": "user"},
            {"role": "assistant", "tool_calls": [{"type": "function"}]},
        ],
    }
    path.write_text(json.dumps(entry) + "\n")
    wl = load_workload(path)
    assert wl.total_tokens_approx == 0


def test_workload_with_malformed_json_line(tmp_path):
    """A line with invalid JSON should raise an error."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"seq": 1}\n{bad json}\n')
    with pytest.raises(json.JSONDecodeError):
        load_workload(path)


def test_workload_path_preserved(tmp_path):
    path = _make_workload_file(tmp_path)
    wl = load_workload(path)
    assert wl.path == str(path)


def test_workload_name_from_stem(tmp_path):
    path = tmp_path / "my-session-recording.jsonl"
    path.write_text(json.dumps({"seq": 1, "messages": []}) + "\n")
    wl = load_workload(path)
    assert wl.name == "my-session-recording"


def test_workload_empty_experiment_ids_excluded():
    """Entries with empty experiment_id should not appear in experiment_ids."""
    wl = Workload(entries=[
        WorkloadEntry(seq=1, experiment_id=""),
        WorkloadEntry(seq=2, experiment_id="exp1"),
    ])
    assert wl.experiment_ids == ["exp1"]


def test_get_workload_by_name_not_found():
    """Looking up a non-existent built-in name should give a clear error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        get_workload("definitely_not_a_real_workload")


# --- Player helpers ---


def test_bucket_label_thresholds():
    """Verify _bucket_label maps token counts to correct profiles."""
    from agentic_swarm_bench.workloads.player import _bucket_label

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


# --- Replay with model_context_length ---


def _make_varied_workload_file(tmp_path):
    """Create a workload with entries of varying sizes for context-length filtering tests."""
    path = tmp_path / "varied.jsonl"
    entries = [
        {
            "seq": 1,
            "experiment_id": "test",
            "messages": [{"role": "user", "content": "x" * 100}],
            "model": "test-model",
            "max_tokens": 64,
        },
        {
            "seq": 2,
            "experiment_id": "test",
            "messages": [{"role": "user", "content": "x" * 80_000}],
            "model": "test-model",
            "max_tokens": 64,
        },
        {
            "seq": 3,
            "experiment_id": "test",
            "messages": [{"role": "user", "content": "x" * 200_000}],
            "model": "test-model",
            "max_tokens": 64,
        },
    ]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)


def test_replay_user_session_skips_oversized_requests(tmp_path):
    """_replay_user_session should skip entries exceeding model_context_length."""
    import asyncio

    from agentic_swarm_bench.workloads.player import _replay_user_session
    from agentic_swarm_bench.workloads.registry import load_workload

    path = _make_varied_workload_file(tmp_path)
    wl = load_workload(path)

    completed = []

    async def run():
        import httpx
        async with httpx.AsyncClient() as client:
            return await _replay_user_session(
                client=client,
                url="http://127.0.0.1:1/unused",
                model_override="test-model",
                headers={},
                entries=wl.entries,
                timeout=1.0,
                user_id=0,
                on_complete=lambda: completed.append(1),
                model_context_length=10_000,
            )

    results = asyncio.run(run())

    assert len(results) == 1
    assert len(completed) == 3


def test_replay_user_session_no_skip_without_limit(tmp_path):
    """Without model_context_length, _replay_user_session attempts all entries."""
    import asyncio

    from agentic_swarm_bench.workloads.player import _replay_user_session
    from agentic_swarm_bench.workloads.registry import load_workload

    path = _make_varied_workload_file(tmp_path)
    wl = load_workload(path)

    completed = []

    async def run():
        import httpx
        async with httpx.AsyncClient() as client:
            return await _replay_user_session(
                client=client,
                url="http://127.0.0.1:1/unused",
                model_override="test-model",
                headers={},
                entries=wl.entries,
                timeout=0.5,
                user_id=0,
                on_complete=lambda: completed.append(1),
                model_context_length=None,
            )

    results = asyncio.run(run())

    assert len(results) == 3
    assert len(completed) == 3
