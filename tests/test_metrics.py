"""Tests for metric collection and statistics."""

import pytest

from agentic_swarm_bench.metrics.collector import (
    BenchmarkRun,
    RequestMetrics,
    ScenarioResult,
    is_context_length_error,
)
from agentic_swarm_bench.metrics.stats import analyze_scenario, compute_distribution


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
    assert m.itl_p50 == 55.0  # true median: (50+60)/2
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


# --- Context length error detection ---


class TestIsContextLengthError:
    """Tests for is_context_length_error across various API error formats."""

    @pytest.mark.parametrize(
        "error_msg",
        [
            'HTTP 400: {"error":{"message":"prompt is too long: 17357 tokens > 4096 maximum"}}',
            'HTTP 400: {"error":{"message":"This model\'s maximum context length is 8192 tokens"}}',
            'HTTP 400: {"error":{"code":"context_length_exceeded"}}',
            'HTTP 400: {"error":"Too many tokens in the input"}}',
            'HTTP 400: {"error":"Input is too long for this model"}}',
            'HTTP 400: {"error":"Request exceeds the model\'s maximum context"}}',
            "HTTP 400: token limit exceeded",
            "HTTP 400: max_prompt_length exceeded",
            "HTTP 400: Please reduce your prompt to fit within the context window",
            "HTTP 400: Maximum allowed tokens exceeded for this model",
        ],
    )
    def test_detects_context_length_errors(self, error_msg):
        assert is_context_length_error(error_msg) is True

    @pytest.mark.parametrize(
        "error_msg",
        [
            "HTTP 400: Bad request",
            "HTTP 401: Unauthorized",
            "HTTP 429: Rate limit exceeded",
            "HTTP 500: Internal server error",
            "ConnectError: Connection refused",
            "TimeoutError: Request timed out",
            "HTTP 400: Invalid model name",
            "HTTP 400: temperature must be between 0 and 2",
        ],
    )
    def test_ignores_non_context_length_errors(self, error_msg):
        assert is_context_length_error(error_msg) is False

    def test_none_returns_false(self):
        assert is_context_length_error(None) is False

    def test_empty_string_returns_false(self):
        assert is_context_length_error("") is False

    def test_case_insensitive(self):
        assert is_context_length_error("HTTP 400: PROMPT IS TOO LONG") is True
        assert is_context_length_error("HTTP 400: Context_Length_Exceeded") is True


# ---------------------------------------------------------------------------
# ITL edge cases
# ---------------------------------------------------------------------------


def test_itl_empty_list_returns_zero():
    m = RequestMetrics(completion_tokens=1)
    assert m.itl_p50 == 0.0
    assert m.itl_p95 == 0.0


def test_itl_single_element():
    m = RequestMetrics(completion_tokens=1, itl_ms=[55.0])
    assert m.itl_p50 == 55.0
    assert m.itl_p95 == 55.0


def test_itl_two_elements():
    m = RequestMetrics(completion_tokens=2, itl_ms=[10.0, 90.0])
    assert m.itl_p50 == 50.0  # true median: (10+90)/2
    assert m.itl_p95 == 90.0


# ---------------------------------------------------------------------------
# to_dict / BenchmarkRun.load roundtrip with thinking tokens
# ---------------------------------------------------------------------------


def test_thinking_roundtrip_via_benchmark_run(tmp_path):
    run = BenchmarkRun(model="think-model", endpoint="http://test:8000", started_at="2026-01-01")
    m = RequestMetrics(
        request_id=1,
        completion_tokens=100,
        thinking_tokens=60,
        ttft_thinking_ms=200.0,
        ttft_visible_ms=5000.0,
        tok_per_sec=30.0,
    )
    scenario = ScenarioResult(num_users=1, wall_time_s=1.0, requests=[m])
    run.scenarios.append(scenario)

    path = str(tmp_path / "thinking_run.json")
    run.save(path)
    loaded = BenchmarkRun.load(path)

    r = loaded.scenarios[0].requests[0]
    assert r.thinking_tokens == 60
    assert r.ttft_thinking_ms == 200.0
    assert r.ttft_visible_ms == 5000.0
    assert r.visible_tokens == 40
    assert r.thinking_overhead_ms == pytest.approx(4800.0)


def test_to_dict_excludes_thinking_fields_when_zero():
    m = RequestMetrics(completion_tokens=50, thinking_tokens=0)
    d = m.to_dict()
    assert "thinking_tokens" not in d
    assert "ttft_thinking_ms" not in d


def test_to_dict_includes_thinking_fields_when_nonzero():
    m = RequestMetrics(
        completion_tokens=100,
        thinking_tokens=30,
        ttft_thinking_ms=100.0,
        ttft_visible_ms=500.0,
    )
    d = m.to_dict()
    assert d["thinking_tokens"] == 30
    assert d["ttft_thinking_ms"] == 100.0
    assert d["ttft_visible_ms"] == 500.0
    assert "thinking_overhead_ms" in d


# ---------------------------------------------------------------------------
# ITL data roundtrip (save/load) and correct median
# ---------------------------------------------------------------------------


def test_itl_roundtrip_via_save_load(tmp_path):
    """Raw itl_ms data must survive a save->load cycle."""
    run = BenchmarkRun(
        model="test", endpoint="http://x", started_at="2026-04-22T00:00:00",
    )
    itl_data = [10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]
    m = RequestMetrics(
        request_id=1, completion_tokens=200, tok_per_sec=44.4, itl_ms=itl_data,
    )
    sr = ScenarioResult(
        num_users=1, context_profile="medium",
        context_tokens=40000, wall_time_s=5.0,
    )
    sr.requests.append(m)
    run.scenarios.append(sr)

    path = str(tmp_path / "itl_run.json")
    run.save(path)
    loaded = BenchmarkRun.load(path)

    m2 = loaded.scenarios[0].requests[0]
    assert m2.itl_ms == itl_data
    assert m2.itl_p50 == 25.0
    assert m2.itl_p95 > 0


def test_itl_ms_serialized_in_to_dict():
    m = RequestMetrics(itl_ms=[1.0, 2.0, 3.0])
    d = m.to_dict()
    assert "itl_ms" in d
    assert d["itl_ms"] == [1.0, 2.0, 3.0]


def test_itl_ms_empty_roundtrip(tmp_path):
    run = BenchmarkRun(model="t", endpoint="e", started_at="t")
    m = RequestMetrics(request_id=1, completion_tokens=10, tok_per_sec=10.0)
    sr = ScenarioResult(num_users=1, wall_time_s=1.0, requests=[m])
    run.scenarios.append(sr)

    path = str(tmp_path / "empty_itl.json")
    run.save(path)
    loaded = BenchmarkRun.load(path)
    assert loaded.scenarios[0].requests[0].itl_ms == []


def test_itl_p50_even_count_is_true_median():
    import statistics

    m = RequestMetrics(itl_ms=[10, 20, 30, 40])
    assert m.itl_p50 == statistics.median([10, 20, 30, 40])
    assert m.itl_p50 == 25.0


def test_itl_p50_six_elements():
    import statistics

    m = RequestMetrics(itl_ms=[5, 10, 15, 20, 25, 30])
    assert m.itl_p50 == statistics.median([5, 10, 15, 20, 25, 30])
    assert m.itl_p50 == 17.5


def test_itl_p50_odd_count():
    m = RequestMetrics(itl_ms=[10, 20, 30])
    assert m.itl_p50 == 20.0
