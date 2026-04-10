"""Tests for metric collection and statistics."""

from agentic_coding_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_coding_bench.metrics.stats import analyze_scenario, compute_distribution


def test_request_metrics_to_dict():
    m = RequestMetrics(
        request_id=1,
        task_id="P1",
        ttft_ms=150.5,
        completion_tokens=42,
        tok_per_sec=28.0,
    )
    d = m.to_dict()
    assert d["request_id"] == 1
    assert d["ttft_ms"] == 150.5
    assert d["tok_per_sec"] == 28.0
    assert d["error"] is None


def test_request_metrics_succeeded():
    m = RequestMetrics(completion_tokens=10)
    assert m.succeeded is True

    m2 = RequestMetrics(completion_tokens=10, error="timeout")
    assert m2.succeeded is False

    m3 = RequestMetrics(completion_tokens=0)
    assert m3.succeeded is False


def test_itl_percentiles():
    m = RequestMetrics(itl_ms=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert m.itl_p50 == 60  # median of sorted list
    assert m.itl_p95 >= 90


def test_compute_distribution_empty():
    stats = compute_distribution([])
    assert stats.count == 0
    assert stats.mean == 0


def test_compute_distribution_values():
    stats = compute_distribution([10, 20, 30, 40, 50])
    assert stats.count == 5
    assert stats.min == 10
    assert stats.max == 50
    assert stats.mean == 30.0
    assert stats.median == 30.0


def test_analyze_scenario():
    reqs = [
        RequestMetrics(
            request_id=i,
            ttft_ms=100 + i * 10,
            completion_tokens=50,
            tok_per_sec=30.0,
            total_time_s=2.0,
        )
        for i in range(5)
    ]
    scenario = ScenarioResult(
        num_users=5,
        context_profile="medium",
        context_tokens=40000,
        wall_time_s=3.0,
        requests=reqs,
    )
    stats = analyze_scenario(scenario)
    assert stats.successful == 5
    assert stats.failed == 0
    assert stats.tok_per_sec.count == 5
    assert stats.aggregate_tok_per_sec > 0


def test_thinking_tokens():
    m = RequestMetrics(
        completion_tokens=100,
        thinking_tokens=60,
        ttft_thinking_ms=200.0,
        ttft_visible_ms=5000.0,
    )
    assert m.visible_tokens == 40
    assert m.thinking_overhead_ms == 4800.0
    d = m.to_dict()
    assert d["thinking_tokens"] == 60
    assert d["thinking_overhead_ms"] == 4800.0


def test_thinking_tokens_zero():
    m = RequestMetrics(completion_tokens=50, thinking_tokens=0)
    assert m.visible_tokens == 50
    assert m.thinking_overhead_ms == 0.0
    d = m.to_dict()
    assert "thinking_tokens" not in d


def test_p99_in_distribution():
    stats = compute_distribution(list(range(1, 101)))
    assert stats.p99 >= 99
    assert stats.p95 >= 95
    assert stats.p5 <= 6


def test_scenario_has_thinking():
    reqs = [
        RequestMetrics(completion_tokens=50, thinking_tokens=20, tok_per_sec=30),
        RequestMetrics(completion_tokens=50, thinking_tokens=0, tok_per_sec=30),
    ]
    scenario = ScenarioResult(num_users=2, wall_time_s=1.0, requests=reqs)
    assert scenario.has_thinking is True
    stats = analyze_scenario(scenario)
    assert stats.has_thinking is True


def test_benchmark_run_serialization(tmp_path):
    run = BenchmarkRun(
        model="test-model",
        endpoint="http://test:8000",
        defeat_cache=True,
        started_at="2026-04-07T00:00:00",
    )
    scenario = ScenarioResult(
        num_users=1,
        context_profile="fresh",
        context_tokens=6000,
        wall_time_s=1.0,
        requests=[RequestMetrics(request_id=1, completion_tokens=10, tok_per_sec=10.0)],
    )
    run.scenarios.append(scenario)

    path = str(tmp_path / "results.json")
    run.save(path)

    loaded = BenchmarkRun.load(path)
    assert loaded.model == "test-model"
    assert len(loaded.scenarios) == 1
    assert len(loaded.scenarios[0].requests) == 1
    assert loaded.scenarios[0].requests[0].tok_per_sec == 10.0
