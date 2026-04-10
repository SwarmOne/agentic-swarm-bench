"""Tests for task loading and filtering."""

from agentic_coding_bench.tasks.registry import (
    filter_tasks,
    get_tasks,
    load_all_tasks,
    parse_task_range,
)


def test_load_all_tasks():
    tasks = load_all_tasks()
    assert len(tasks) == 110


def test_task_ids_are_sequential():
    tasks = load_all_tasks()
    for i, task in enumerate(tasks, start=1):
        assert task["id"] == f"P{i}"


def test_all_tiers_present():
    tasks = load_all_tasks()
    tiers = {t["tier"] for t in tasks}
    assert tiers == {"trivial", "easy", "medium", "hard", "expert"}


def test_tier_counts():
    tasks = load_all_tasks()
    counts = {}
    for t in tasks:
        counts[t["tier"]] = counts.get(t["tier"], 0) + 1
    assert counts["trivial"] == 11
    assert counts["easy"] == 17
    assert counts["medium"] == 28
    assert counts["hard"] == 27
    assert counts["expert"] == 27


def test_every_task_has_required_fields():
    tasks = load_all_tasks()
    for t in tasks:
        assert "id" in t
        assert "tier" in t
        assert "prompt" in t
        assert "tags" in t
        assert "max_output_tokens" in t
        assert len(t["prompt"]) > 10


def test_parse_task_range_p_notation():
    assert parse_task_range("p1-p25") == (1, 25)
    assert parse_task_range("P51-P75") == (51, 75)


def test_parse_task_range_numeric():
    assert parse_task_range("1-50") == (1, 50)


def test_parse_task_range_single():
    assert parse_task_range("p10") == (10, 10)


def test_parse_task_range_tier_name():
    assert parse_task_range("trivial") == (1, 10)
    assert parse_task_range("expert") == (76, 100)


def test_filter_by_range():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, task_range="p1-p10")
    assert len(filtered) == 10
    assert all(t["tier"] == "trivial" for t in filtered)


def test_filter_by_tier():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, tier="hard")
    assert len(filtered) == 27


def test_filter_by_tags():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, tags=["basics"])
    assert len(filtered) >= 5
    assert all("basics" in t["tags"] for t in filtered)


def test_filter_by_language_tags():
    tasks = load_all_tasks()
    ts_tasks = filter_tasks(tasks, tags=["typescript"])
    assert len(ts_tasks) >= 3
    rust_tasks = filter_tasks(tasks, tags=["rust"])
    assert len(rust_tasks) >= 3
    go_tasks = filter_tasks(tasks, tags=["go"])
    assert len(go_tasks) >= 2


def test_get_tasks_combined():
    result = get_tasks(task_range="p1-p25", tier="easy")
    for t in result:
        assert t["tier"] == "easy"
        num = int(t["id"][1:])
        assert 1 <= num <= 25
