"""Tests for report generation."""

from agentic_coding_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_coding_bench.metrics.stats import DistributionStats, ScenarioStats
from agentic_coding_bench.report.markdown import (
    _base_profile,
    _experience_label,
    _grade,
    _grade_icon,
    _verdict_for_stats,
    _verdict_label,
    generate_comparison,
    generate_report,
)


def _make_run(model="test-model", endpoint="http://test:8000"):
    run = BenchmarkRun(
        model=model,
        endpoint=endpoint,
        defeat_cache=True,
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


def _make_multi_profile_run():
    """Create a run with multiple context profiles for testing scaling sections."""
    run = BenchmarkRun(
        model="test-model",
        endpoint="http://test:8000",
        defeat_cache=True,
        started_at="2026-04-07T12:00:00",
    )
    profiles = [
        ("fresh", 6000, 50, 35.0),
        ("medium", 40000, 200, 25.0),
        ("long", 70000, 500, 18.0),
    ]
    for profile, tokens, ttft, toks in profiles:
        reqs = [
            RequestMetrics(
                request_id=0,
                user_id=0,
                task_id="P1",
                context_profile=profile,
                context_tokens=tokens,
                ttft_ms=ttft,
                total_time_s=2.0,
                completion_tokens=50,
                tok_per_sec=toks,
                prefill_tok_per_sec=tokens / (ttft / 1000),
            )
        ]
        run.scenarios.append(
            ScenarioResult(
                num_users=1,
                context_profile=profile,
                context_tokens=tokens,
                wall_time_s=2.5,
                requests=reqs,
            )
        )
    return run


def _make_empty_run():
    """A run with no scenarios at all."""
    return BenchmarkRun(
        model="test-model",
        endpoint="http://test:8000",
        defeat_cache=True,
        started_at="2026-04-07T12:00:00",
    )


def _make_all_failures_run():
    """A run where every request failed."""
    run = BenchmarkRun(
        model="fail-model",
        endpoint="http://test:8000",
        defeat_cache=True,
        started_at="2026-04-07T12:00:00",
    )
    reqs = [
        RequestMetrics(
            request_id=0,
            user_id=0,
            task_id="P1",
            context_profile="medium",
            context_tokens=40000,
            error="HTTP 500: Internal Server Error",
        )
    ]
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="medium",
            context_tokens=40000,
            wall_time_s=1.0,
            requests=reqs,
        )
    )
    return run


def _make_thinking_run():
    """A run with reasoning/thinking tokens."""
    run = BenchmarkRun(
        model="reasoning-model",
        endpoint="http://test:8000",
        defeat_cache=True,
        started_at="2026-04-07T12:00:00",
    )
    reqs = [
        RequestMetrics(
            request_id=0,
            user_id=0,
            task_id="P1",
            context_profile="medium",
            context_tokens=40000,
            ttft_ms=200,
            total_time_s=3.0,
            completion_tokens=80,
            tok_per_sec=25.0,
            prefill_tok_per_sec=40000.0,
            thinking_tokens=30,
            ttft_thinking_ms=100,
            ttft_visible_ms=500,
        )
    ]
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="medium",
            context_tokens=40000,
            wall_time_s=3.0,
            requests=reqs,
        )
    )
    return run


# --- Basic report tests ---


def test_generate_report_contains_branding():
    report = generate_report(_make_run())
    assert "SwarmOne" in report
    assert "swarmone-logo" in report


def test_generate_report_contains_model():
    report = generate_report(_make_run(model="qwen-27b"))
    assert "qwen-27b" in report


def test_generate_report_contains_metrics():
    report = generate_report(_make_run())
    assert "TTFT" in report
    assert "Tok/s" in report
    assert "Prefill" in report


def test_generate_report_contains_methodology():
    report = generate_report(_make_run())
    assert "Methodology" in report
    assert "prefix cache" in report.lower() or "Prefix" in report


def test_generate_report_contains_verdict():
    report = generate_report(_make_run())
    assert "Verdict" in report
    assert "agentic coding" in report.lower()


def test_generate_report_contains_key_findings():
    report = generate_report(_make_run())
    assert "Key Findings" in report


def test_generate_report_contains_experience_column():
    report = generate_report(_make_run())
    assert "Experience" in report
    assert "streaming" in report


def test_generate_report_contains_performance_grades():
    report = generate_report(_make_run())
    assert "Performance grades" in report


def test_generate_report_context_scaling():
    report = generate_report(_make_multi_profile_run())
    assert "Context Scaling" in report


def test_generate_report_grade_icons():
    report = generate_report(_make_run())
    has_icon = any(icon in report for icon in ["🟢", "🟡", "🔴"])
    assert has_icon


def test_generate_comparison():
    run_a = _make_run(model="model-a")
    run_b = _make_run(model="model-b")
    report = generate_comparison(run_a, run_b)
    assert "model-a" in report
    assert "model-b" in report
    assert "Comparison" in report


def test_generate_comparison_has_winner():
    run_a = _make_run(model="model-a")
    run_b = _make_run(model="model-b")
    report = generate_comparison(run_a, run_b)
    assert "Tied" in report or "wins" in report


# --- Edge case: empty run ---


def test_generate_report_empty_run():
    report = generate_report(_make_empty_run())
    assert "Verdict" in report
    assert "No successful requests" in report


def test_generate_report_empty_run_no_crash():
    """An empty run should not raise any exceptions."""
    report = generate_report(_make_empty_run())
    assert isinstance(report, str)
    assert len(report) > 0


# --- Edge case: all failures ---


def test_generate_report_all_failures_verdict():
    report = generate_report(_make_all_failures_run())
    assert "No successful requests" in report


def test_generate_report_all_failures_shows_fail():
    report = generate_report(_make_all_failures_run())
    assert "FAIL" in report


def test_generate_report_all_failures_no_key_findings():
    report = generate_report(_make_all_failures_run())
    assert "Key Findings" not in report


def test_generate_report_all_failures_no_experience():
    report = generate_report(_make_all_failures_run())
    assert "streaming" not in report


# --- Edge case: single profile skips per-profile section ---


def test_results_table_has_completed_column():
    report = generate_report(_make_run())
    assert "Completed" in report
    assert "Status" not in report.split("Methodology")[0]


def test_results_table_has_output_tok_column():
    report = generate_report(_make_run())
    assert "Output tok" in report


# --- Edge case: thinking tokens ---


def test_thinking_run_shows_reasoning_section():
    report = generate_report(_make_thinking_run())
    assert "Reasoning Token Analysis" in report


def test_thinking_run_key_findings_mentions_overhead():
    report = generate_report(_make_thinking_run())
    assert "thinking overhead" in report.lower() or "Thinking" in report


# --- Edge case: context scaling with only 1 profile ---


def test_context_scaling_skipped_with_one_profile():
    run = _make_run()
    report = generate_report(run)
    assert "Context Scaling" not in report


# --- Edge case: context scaling chart correctness ---


def test_context_scaling_chart_has_both_bars():
    report = generate_report(_make_multi_profile_run())
    assert "▓" in report
    assert "█" in report


# --- Edge case: concurrency scaling ---


def test_concurrency_scaling_section():
    run = _make_run()
    report = generate_report(run)
    assert "Concurrency Scaling" in report
    assert "Efficiency" in report


# --- Grading helper tests ---


class TestGrade:
    def test_lower_is_better_good(self):
        assert _grade(100, 3000, 10000, lower_is_better=True) == "good"

    def test_lower_is_better_ok(self):
        assert _grade(5000, 3000, 10000, lower_is_better=True) == "ok"

    def test_lower_is_better_poor(self):
        assert _grade(15000, 3000, 10000, lower_is_better=True) == "poor"

    def test_lower_is_better_boundary_good(self):
        assert _grade(3000, 3000, 10000, lower_is_better=True) == "good"

    def test_lower_is_better_boundary_ok(self):
        assert _grade(10000, 3000, 10000, lower_is_better=True) == "ok"

    def test_higher_is_better_good(self):
        assert _grade(50, 30, 15, lower_is_better=False) == "good"

    def test_higher_is_better_ok(self):
        assert _grade(20, 30, 15, lower_is_better=False) == "ok"

    def test_higher_is_better_poor(self):
        assert _grade(10, 30, 15, lower_is_better=False) == "poor"

    def test_higher_is_better_boundary_good(self):
        assert _grade(30, 30, 15, lower_is_better=False) == "good"

    def test_higher_is_better_boundary_ok(self):
        assert _grade(15, 30, 15, lower_is_better=False) == "ok"

    def test_zero_value_lower_is_better(self):
        assert _grade(0, 3000, 10000, lower_is_better=True) == "good"

    def test_zero_value_higher_is_better(self):
        assert _grade(0, 30, 15, lower_is_better=False) == "poor"

    def test_negative_value(self):
        assert _grade(-1, 3000, 10000, lower_is_better=True) == "good"


class TestGradeIcon:
    def test_good(self):
        assert _grade_icon("good") == "🟢"

    def test_ok(self):
        assert _grade_icon("ok") == "🟡"

    def test_poor(self):
        assert _grade_icon("poor") == "🔴"

    def test_unknown(self):
        assert _grade_icon("unknown") == "⚪"

    def test_empty(self):
        assert _grade_icon("") == "⚪"


class TestVerdictForStats:
    def test_good_verdict(self):
        stats = ScenarioStats(
            successful=1,
            ttft_ms=DistributionStats(median=500),
            tok_per_sec=DistributionStats(median=40),
        )
        assert _verdict_for_stats(stats) == "good"

    def test_poor_verdict_from_ttft(self):
        stats = ScenarioStats(
            successful=1,
            ttft_ms=DistributionStats(median=15000),
            tok_per_sec=DistributionStats(median=40),
        )
        assert _verdict_for_stats(stats) == "poor"

    def test_poor_verdict_from_toks(self):
        stats = ScenarioStats(
            successful=1,
            ttft_ms=DistributionStats(median=500),
            tok_per_sec=DistributionStats(median=5),
        )
        assert _verdict_for_stats(stats) == "poor"

    def test_ok_verdict(self):
        stats = ScenarioStats(
            successful=1,
            ttft_ms=DistributionStats(median=5000),
            tok_per_sec=DistributionStats(median=40),
        )
        assert _verdict_for_stats(stats) == "ok"

    def test_worst_grade_wins(self):
        """If TTFT is good but tok/s is poor, overall should be poor."""
        stats = ScenarioStats(
            successful=1,
            ttft_ms=DistributionStats(median=100),
            tok_per_sec=DistributionStats(median=5),
        )
        assert _verdict_for_stats(stats) == "poor"


class TestVerdictLabel:
    def test_good(self):
        assert "GOOD" in _verdict_label("good")

    def test_ok(self):
        assert "MARGINAL" in _verdict_label("ok")

    def test_poor(self):
        assert "POOR" in _verdict_label("poor")

    def test_unknown(self):
        assert "UNKNOWN" in _verdict_label("nonexistent")


class TestBaseProfile:
    def test_plain(self):
        assert _base_profile("medium") == "medium"

    def test_with_cache_label(self):
        assert _base_profile("medium (cold)") == "medium"

    def test_with_warm_label(self):
        assert _base_profile("long (warm)") == "long"

    def test_no_parens(self):
        assert _base_profile("fresh") == "fresh"

    def test_empty(self):
        assert _base_profile("") == ""

    def test_nested_parens(self):
        assert _base_profile("medium (cold) (extra)") == "medium"


# --- Comparison edge cases ---


def test_comparison_empty_runs():
    run_a = _make_empty_run()
    run_b = _make_empty_run()
    report = generate_comparison(run_a, run_b)
    assert "No comparable scenarios" in report


def test_comparison_one_faster():
    """Candidate with higher tok/s should win."""
    run_a = _make_run(model="slow")
    run_b = _make_run(model="fast")
    for s in run_b.scenarios:
        for r in s.requests:
            r.tok_per_sec = 50.0
    report = generate_comparison(run_a, run_b)
    assert "fast" in report
    assert "wins" in report


def test_comparison_mixed_scenarios():
    """Comparison where runs have different scenario sets."""
    run_a = _make_run(model="a")
    run_b = BenchmarkRun(
        model="b", endpoint="http://test:8000",
        defeat_cache=True, started_at="2026-04-07T12:00:00",
    )
    run_b.scenarios.append(
        ScenarioResult(
            num_users=1, context_profile="long", context_tokens=70000,
            wall_time_s=2.5,
            requests=[
                RequestMetrics(
                    request_id=0, user_id=0, task_id="P1",
                    context_profile="long", context_tokens=70000,
                    ttft_ms=300, total_time_s=2.0,
                    completion_tokens=50, tok_per_sec=30.0,
                )
            ],
        )
    )
    report = generate_comparison(run_a, run_b)
    assert "N/A" in report or "0.0" in report


# --- Report with no started_at timestamp ---


def test_report_missing_timestamp():
    run = _make_run()
    run.started_at = ""
    report = generate_report(run)
    assert "N/A" in report


# --- Report with defeat_cache disabled ---


def test_report_without_cache_defeat():
    run = _make_run()
    run.defeat_cache = False
    report = generate_report(run)
    assert "Disabled" in report
    assert "unique random salt" not in report


def test_report_with_cache_defeat():
    run = _make_run()
    run.defeat_cache = True
    report = generate_report(run)
    assert "Enabled" in report
    assert "unique random salt" in report


# --- Experience label tests ---


class TestExperienceLabel:
    def test_instant_fast(self):
        label = _experience_label(500, 60)
        assert "Instant" in label
        assert "fast streaming" in label

    def test_responsive_smooth(self):
        label = _experience_label(2000, 35)
        assert "Responsive" in label
        assert "smooth streaming" in label

    def test_noticeable_slow(self):
        label = _experience_label(7000, 20)
        assert "Noticeable wait" in label
        assert "slow streaming" in label

    def test_disruptive_sluggish(self):
        label = _experience_label(12000, 10)
        assert "Disruptive" in label
        assert "sluggish" in label

    def test_slight_pause(self):
        label = _experience_label(4000, 60)
        assert "Slight pause" in label
