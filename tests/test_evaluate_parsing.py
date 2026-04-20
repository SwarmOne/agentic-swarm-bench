"""Tests for evaluate field parsing in scenario registry."""

from __future__ import annotations

import json

from agentic_swarm_bench.scenarios.registry import (
    Scenario,
    Task,
    get_scenario,
    load_scenario,
)


def _write_recording(path, n=1):
    """Write a minimal JSONL recording."""
    with open(path, "w") as f:
        for i in range(n):
            entry = {"seq": i + 1, "messages": [{"role": "user", "content": f"Q{i}"}]}
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Parsing from directory manifests
# ---------------------------------------------------------------------------


def test_evaluate_parsed_from_directory_manifest(tmp_path):
    scenario_dir = tmp_path / "eval-test"
    scenario_dir.mkdir()
    _write_recording(scenario_dir / "t.jsonl")

    manifest = {
        "name": "eval-test",
        "tasks": [
            {
                "id": "t",
                "recording": "t.jsonl",
                "evaluate": [{"type": "contains", "value": "Paris"}],
            }
        ],
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    s = load_scenario(scenario_dir)
    assert s.tasks[0].evaluate is not None
    assert len(s.tasks[0].evaluate) == 1
    assert s.tasks[0].evaluate[0]["type"] == "contains"
    assert s.tasks[0].evaluate[0]["value"] == "Paris"


def test_evaluate_none_when_absent(tmp_path):
    scenario_dir = tmp_path / "no-eval"
    scenario_dir.mkdir()
    _write_recording(scenario_dir / "t.jsonl")

    manifest = {
        "name": "no-eval",
        "tasks": [{"id": "t", "recording": "t.jsonl"}],
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    s = load_scenario(scenario_dir)
    assert s.tasks[0].evaluate is None


# ---------------------------------------------------------------------------
# Parsing from standalone JSON files
# ---------------------------------------------------------------------------


def test_evaluate_parsed_from_standalone_json(tmp_path):
    _write_recording(tmp_path / "t.jsonl")

    manifest = {
        "name": "standalone-eval",
        "tasks": [
            {
                "id": "t",
                "recording": "t.jsonl",
                "evaluate": [
                    {"type": "contains", "value": "Jupiter"},
                    {"type": "regex", "pattern": r"\d+"},
                ],
            }
        ],
    }
    path = tmp_path / "standalone.json"
    path.write_text(json.dumps(manifest))

    s = load_scenario(path)
    assert len(s.tasks[0].evaluate) == 2
    assert s.tasks[0].evaluate[0]["type"] == "contains"
    assert s.tasks[0].evaluate[1]["type"] == "regex"


# ---------------------------------------------------------------------------
# Multiple directives and seq targeting
# ---------------------------------------------------------------------------


def test_evaluate_with_seq_field(tmp_path):
    scenario_dir = tmp_path / "seq-test"
    scenario_dir.mkdir()
    _write_recording(scenario_dir / "t.jsonl", n=3)

    manifest = {
        "name": "seq-test",
        "tasks": [
            {
                "id": "t",
                "recording": "t.jsonl",
                "evaluate": [
                    {"type": "contains", "value": "answer", "seq": 3},
                ],
            }
        ],
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    s = load_scenario(scenario_dir)
    assert s.tasks[0].evaluate[0]["seq"] == 3


def test_evaluate_with_case_sensitive_flag(tmp_path):
    _write_recording(tmp_path / "t.jsonl")

    manifest = {
        "name": "case-test",
        "tasks": [
            {
                "id": "t",
                "recording": "t.jsonl",
                "evaluate": [
                    {"type": "contains", "value": "paris", "case_sensitive": False},
                ],
            }
        ],
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(manifest))

    s = load_scenario(path)
    assert s.tasks[0].evaluate[0]["case_sensitive"] is False


# ---------------------------------------------------------------------------
# has_evaluations property
# ---------------------------------------------------------------------------


def test_has_evaluations_true_when_any_task_has_evaluate():
    s = Scenario(
        name="test",
        tasks=[
            Task(id="t1", evaluate=None),
            Task(id="t2", evaluate=[{"type": "contains", "value": "yes"}]),
        ],
    )
    assert s.has_evaluations is True


def test_has_evaluations_false_when_no_task_has_evaluate():
    s = Scenario(
        name="test",
        tasks=[
            Task(id="t1", evaluate=None),
            Task(id="t2", evaluate=None),
        ],
    )
    assert s.has_evaluations is False


def test_has_evaluations_false_with_empty_list():
    s = Scenario(
        name="test",
        tasks=[Task(id="t1", evaluate=[])],
    )
    assert s.has_evaluations is False


def test_has_evaluations_false_no_tasks():
    s = Scenario(name="test", tasks=[])
    assert s.has_evaluations is False


# ---------------------------------------------------------------------------
# Built-in trivial-qa now has evaluate directives
# ---------------------------------------------------------------------------


def test_trivial_qa_has_evaluate_directives():
    s = get_scenario("trivial-qa")
    assert s.has_evaluations is True
    for task in s.tasks:
        assert task.evaluate is not None
        assert len(task.evaluate) >= 1


def test_trivial_qa_evaluate_types():
    """Verify the specific evaluate directives we added."""
    s = get_scenario("trivial-qa")
    task_map = {t.id: t for t in s.tasks}

    paris = task_map["capital-of-france"]
    assert paris.evaluate[0]["type"] == "contains"
    assert paris.evaluate[0]["value"] == "Paris"

    speed = task_map["speed-of-light"]
    assert speed.evaluate[0]["type"] == "regex"


# ---------------------------------------------------------------------------
# evaluate preserved through task filter
# ---------------------------------------------------------------------------


def test_evaluate_preserved_through_task_filter(tmp_path):
    scenario_dir = tmp_path / "filter-eval"
    scenario_dir.mkdir()
    _write_recording(scenario_dir / "a.jsonl")
    _write_recording(scenario_dir / "b.jsonl")

    manifest = {
        "name": "filter-eval",
        "tasks": [
            {
                "id": "a",
                "recording": "a.jsonl",
                "evaluate": [{"type": "contains", "value": "alpha"}],
            },
            {
                "id": "b",
                "recording": "b.jsonl",
                "evaluate": [{"type": "contains", "value": "beta"}],
            },
        ],
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest))

    s = get_scenario(str(scenario_dir), task_filter="a")
    assert len(s.tasks) == 1
    assert s.tasks[0].evaluate[0]["value"] == "alpha"


# ---------------------------------------------------------------------------
# JSONL loading (no evaluate support)
# ---------------------------------------------------------------------------


def test_jsonl_loading_has_no_evaluate(tmp_path):
    _write_recording(tmp_path / "plain.jsonl")
    s = load_scenario(tmp_path / "plain.jsonl")
    assert s.tasks[0].evaluate is None
