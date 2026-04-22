"""Regression tests for bugs found during hands-on testing.

Each test pins one specific failure mode so a future regression flips a red
bar in CI instead of slipping through.

Early bugs (1-4) were found while building the CLI:
  1. list-tasks/list-scenarios --format json emitted invalid JSON
     (Rich console.print wraps lines, inserting bare newlines inside strings).
  2. asb compare showed "Tied" when one side had zero successful requests.
  3. ASB_ENDPOINT/ASB_MODEL env vars and YAML endpoint/model not accepted
     (Click required=True blocked them before build_config could inject them).
  4. asb list-tasks -t p999 silently returned all 110 tasks instead of erroring.

Later bugs (5-24) were found during a full endpoint-verdict QA pass on v3.1.0:
  5.  Replay --slice-tokens ignored (used stale recorded prompt_tokens).
  6.  Speed -r / --repetitions silently ignored.
  7.  asb record double-appended /v1/chat/completions to the upstream URL.
  8.  asb speed exited 0 when every request failed.
  9.  --max-tokens was shadowed by per-task max_output_tokens instead of capping it.
  10. No way to pass vendor-specific body (e.g. Qwen enable_thinking).
  11. JSON output lacked a top-level verdict/summary block for CI pipelines.
  12. Ctrl+C wiped completed scenarios + any mid-flight requests that finished.
  13. -u 0, negative timeouts, and other nonsense values were accepted.
  14. cache_mode was not stored as its own field on ScenarioResult.
  15. cache-mode realistic report showed one combined verdict instead of per-pass.
  16. Error summary lacked error-kind breakdown (ConnectionError vs HTTP 401 etc).
  17. request_id collided across scenarios / repetitions.
  18. asb compare misreported "no overlap" as "all failed".
  19. --help Modes list omitted record/replay/list-scenarios/report/compare.
  20. ITL sub-millisecond rendered as 0ms instead of <1ms.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest
from click.testing import CliRunner

from agentic_swarm_bench.cli import main
from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.metrics.collector import BenchmarkRun, RequestMetrics, ScenarioResult
from agentic_swarm_bench.report.markdown import generate_comparison
from agentic_swarm_bench.runner.direct import _fmt_ms
from agentic_swarm_bench.scenarios.player import _entry_prompt_tokens, _slice_entries
from agentic_swarm_bench.scenarios.recorder import _normalize_recorder_upstream

# ---------------------------------------------------------------------------
# Bug 1 – JSON output is valid JSON
# ---------------------------------------------------------------------------


def test_list_tasks_json_is_valid():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--format", "json"])
    assert result.exit_code == 0, result.output
    tasks = json.loads(result.output)
    assert isinstance(tasks, list)
    assert len(tasks) == 110
    assert tasks[0]["id"] == "P1"


def test_list_scenarios_json_is_valid():
    runner = CliRunner()
    result = runner.invoke(main, ["list-scenarios", "--format", "json"])
    assert result.exit_code == 0, result.output
    scenarios = json.loads(result.output)
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 2
    names = [s["name"] for s in scenarios]
    assert "js-coding-opus" in names
    assert "trivial-qa" in names


def test_list_tasks_json_no_embedded_newlines():
    """The specific failure mode: Rich wraps long strings with bare newlines."""
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--format", "json"])
    assert result.exit_code == 0
    # If any bare newline appears inside a JSON string value, json.loads raises
    parsed = json.loads(result.output)
    assert len(parsed) > 0


# ---------------------------------------------------------------------------
# Bug 2 – compare winner summary with one-sided failures
# ---------------------------------------------------------------------------


def _make_run(model: str, tok_per_sec: float, profile: str = "fresh", ok: int = 5):
    run = BenchmarkRun(model=model, endpoint="http://test", started_at="2026-01-01T00:00:00")
    reqs = []
    for i in range(ok):
        reqs.append(
            RequestMetrics(
                request_id=i,
                user_id=0,
                task_id=f"P{i+1}",
                context_profile=profile,
                context_tokens=6000,
                ttft_ms=500.0,
                total_time_s=1.0,
                completion_tokens=50,
                tok_per_sec=tok_per_sec,
                prefill_tok_per_sec=5000.0,
            )
        )
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile=profile,
            context_tokens=6000,
            wall_time_s=5.0,
            requests=reqs,
        )
    )
    return run


def _make_failed_run(model: str, profile: str = "fresh"):
    """A run where all requests failed (error field set, no metrics)."""
    run = BenchmarkRun(model=model, endpoint="http://test", started_at="2026-01-01T00:00:00")
    reqs = [
        RequestMetrics(
            request_id=0,
            user_id=0,
            task_id="P1",
            context_profile=profile,
            context_tokens=6000,
            error="HTTP 500: Internal Server Error",
        )
    ]
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile=profile,
            context_tokens=6000,
            wall_time_s=1.0,
            requests=reqs,
        )
    )
    return run


def test_compare_candidate_full_failure_not_tied():
    """When candidate has zero successes, baseline should win - not 'Tied'."""
    baseline = _make_run("fast-model", tok_per_sec=60.0, profile="fresh")
    candidate = _make_failed_run("broken-model", profile="fresh")

    report = generate_comparison(baseline, candidate)

    assert "Tied" not in report
    assert "Baseline" in report  # baseline wins


def test_compare_one_sided_failure_excluded_from_count():
    """Scenario with 0 successes on one side is excluded from the win count."""
    # baseline wins fresh, candidate fails on short
    baseline_fresh = _make_run("model-a", tok_per_sec=50.0, profile="fresh")
    baseline_fresh.scenarios.append(
        _make_run("model-a", tok_per_sec=45.0, profile="short").scenarios[0]
    )
    candidate_fresh = _make_run("model-b", tok_per_sec=30.0, profile="fresh")
    failed_short = _make_failed_run("model-b", profile="short")
    candidate_fresh.scenarios.append(failed_short.scenarios[0])

    report = generate_comparison(baseline_fresh, candidate_fresh)

    # baseline wins fresh (50 > 30), short is excluded (candidate failed)
    # so baseline wins 1/1 valid scenario, short is noted as excluded
    assert "Baseline" in report
    assert "excluded" in report or "zero completions" in report
    assert "Tied" not in report


def test_compare_both_fail_no_valid_scenarios():
    """If both sides fail on all scenarios, report no valid comparisons."""
    run_a = _make_failed_run("model-a")
    run_b = _make_failed_run("model-b")

    report = generate_comparison(run_a, run_b)

    assert "No valid comparison scenarios" in report
    assert "Tied" not in report


def test_compare_honest_tie_still_works():
    """A genuine tie (both succeed, same tok/s) still says Tied."""
    run_a = _make_run("model-a", tok_per_sec=50.0)
    run_b = _make_run("model-b", tok_per_sec=50.0)

    report = generate_comparison(run_a, run_b)

    assert "Tied" in report


# ---------------------------------------------------------------------------
# Bug 3 – ASB_ENDPOINT / ASB_MODEL env vars respected by CLI
# ---------------------------------------------------------------------------


def test_speed_env_vars_accepted(monkeypatch):
    """ASB_ENDPOINT and ASB_MODEL env vars should satisfy the endpoint/model requirement."""
    monkeypatch.setenv("ASB_ENDPOINT", "http://env-server:8000")
    monkeypatch.setenv("ASB_MODEL", "env-model")
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "--dry-run", "-t", "p1-p3", "-u", "1", "-p", "fresh"])
    assert result.exit_code == 0, result.output
    assert "env-model" in result.output
    assert "env-server" in result.output


def test_speed_missing_endpoint_shows_helpful_error():
    """Without endpoint on CLI or in env, show a clear error (not Click's generic one)."""
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "-m", "some-model", "--dry-run", "-t", "p1"])
    assert result.exit_code != 0
    assert "endpoint" in result.output.lower()


def test_speed_missing_model_shows_helpful_error():
    """Without model on CLI or in env, show a clear error."""
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "-e", "http://server:8000", "--dry-run", "-t", "p1"])
    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_yaml_config_endpoint_model_accepted(tmp_path):
    """endpoint and model from a YAML config file should satisfy the requirement."""
    config_file = tmp_path / "bench.yml"
    config_file.write_text("endpoint: http://yaml-server:9000\nmodel: yaml-model\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--config", str(config_file), "speed", "--dry-run", "-t", "p1", "-u", "1", "-p", "fresh"],
    )
    assert result.exit_code == 0, result.output
    assert "yaml-model" in result.output
    assert "yaml-server" in result.output


# ---------------------------------------------------------------------------
# Bug 4 – Invalid task range raises UsageError (not silent fallback)
# ---------------------------------------------------------------------------


def test_list_tasks_invalid_range_errors():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "-t", "p999"])
    assert result.exit_code != 0
    assert "No tasks matched" in result.output


def test_list_tasks_invalid_tags_errors():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "--tags", "nonexistent_lang_xyz"])
    assert result.exit_code != 0
    assert "No tasks matched" in result.output


def test_list_tasks_no_filter_still_works():
    """list-tasks with no filter should still return all 110 tasks (no regression)."""
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks"])
    assert result.exit_code == 0
    assert "110 total" in result.output


def test_list_tasks_valid_range_works():
    runner = CliRunner()
    result = runner.invoke(main, ["list-tasks", "-t", "p1-p10"])
    assert result.exit_code == 0
    assert "10 total" in result.output


# ---------------------------------------------------------------------------
# Bug 5 – --slice-tokens uses max(recorded, estimated) to survive stale
# prompt_tokens in recordings
# ---------------------------------------------------------------------------


class _FakeEntry:
    """Minimal stand-in for RecordingEntry used by the player slicer."""

    def __init__(self, messages, prompt_tokens=None):
        self.messages = messages
        self.prompt_tokens = prompt_tokens


def test_entry_prompt_tokens_prefers_char_estimate_when_recorded_is_stale():
    # The recorder often writes prompt_tokens=1 for streaming responses, so a
    # naive slicer would sneak a 200K-token recording past an 8K budget.
    msgs = [{"role": "user", "content": "x" * 40_000}]
    entry = _FakeEntry(msgs, prompt_tokens=1)
    tokens = _entry_prompt_tokens(entry)
    assert tokens >= 9_000  # 40_000 chars / 4 ≈ 10K tokens


def test_entry_prompt_tokens_uses_recorded_when_larger():
    msgs = [{"role": "user", "content": "hi"}]
    entry = _FakeEntry(msgs, prompt_tokens=500)
    assert _entry_prompt_tokens(entry) == 500


def test_slice_entries_respects_budget_with_stale_recorded_tokens():
    entries = [_FakeEntry([{"role": "user", "content": "x" * 40_000}], prompt_tokens=1)] * 10
    sliced = _slice_entries(entries, slice_tokens=8_000)
    assert len(sliced) <= 1, (
        f"slicer kept too many entries: {len(sliced)} at ~10K tokens each"
    )


# ---------------------------------------------------------------------------
# Bug 6 – -r / --repetitions multiplies samples on asb speed
# ---------------------------------------------------------------------------


def test_speed_repetitions_accepted_in_dry_run():
    # Dry-run exercises the CLI wiring without touching the network.
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["speed", "-e", "http://test:8000", "-m", "m", "--dry-run",
         "-p", "fresh", "-u", "2", "-r", "5"],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test_run_scenario_repetitions_multiplies_requests(monkeypatch):
    from agentic_swarm_bench.runner import direct as runner_module

    captured: list[tuple[int, int]] = []

    async def fake_send(**kwargs):
        captured.append((kwargs["request_id"], kwargs["repetition_id"]))
        on_complete = kwargs.get("on_complete")
        if on_complete:
            on_complete()
        return RequestMetrics(
            request_id=kwargs["request_id"],
            repetition_id=kwargs["repetition_id"],
            user_id=kwargs["user_id"],
            task_id=kwargs["task_id"],
            context_profile=kwargs["context_profile"],
            context_tokens=kwargs["context_tokens"],
            ttft_ms=100.0,
            total_time_s=1.0,
            completion_tokens=10,
            tok_per_sec=10.0,
        )

    monkeypatch.setattr(runner_module, "_send_streaming_request", fake_send)

    config = BenchmarkConfig(endpoint="http://x", model="m", max_output_tokens=16)
    tasks = [{"id": "P1", "prompt": "hi", "max_output_tokens": 16}]

    scenario = await runner_module.run_scenario(
        config=config,
        url="http://x/v1/chat/completions",
        headers={},
        num_users=3,
        context_tokens=6000,
        context_profile="fresh",
        tasks=tasks,
        repetitions=4,
        defeat_cache=False,
    )

    assert len(scenario.requests) == 12  # 3 users × 4 reps
    rep_ids = sorted({r for _, r in captured})
    assert rep_ids == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Bug 7 – asb record no longer double-appends /v1/chat/completions
# ---------------------------------------------------------------------------


def test_recorder_upstream_normalization_strips_known_suffixes():
    assert _normalize_recorder_upstream("http://u:8000") == "http://u:8000"
    assert (
        _normalize_recorder_upstream("http://u:8000/v1/chat/completions")
        == "http://u:8000"
    )
    assert (
        _normalize_recorder_upstream("http://u:8000/v1/chat/completions/")
        == "http://u:8000"
    )
    assert (
        _normalize_recorder_upstream("https://api.swarmone.ai/K/v1/chat/completions")
        == "https://api.swarmone.ai/K"
    )
    assert (
        _normalize_recorder_upstream("https://anthro.local/v1/messages")
        == "https://anthro.local"
    )


# ---------------------------------------------------------------------------
# Bug 8 – asb speed exits non-zero when every request fails or when
# endpoint/model is missing
# ---------------------------------------------------------------------------


def test_speed_missing_endpoint_exits_nonzero(monkeypatch):
    # Clear env so the test is stable regardless of the developer's shell.
    for var in ("ASB_ENDPOINT", "ASB_MODEL", "ASB_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["speed", "-m", "m", "-p", "fresh", "-u", "1"])
    assert result.exit_code != 0


def test_speed_missing_model_exits_nonzero(monkeypatch):
    for var in ("ASB_ENDPOINT", "ASB_MODEL", "ASB_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runner = CliRunner()
    result = runner.invoke(
        main, ["speed", "-e", "http://test:8000", "-p", "fresh", "-u", "1"]
    )
    assert result.exit_code != 0


def test_enforce_exit_code_raises_when_all_requests_failed():
    from agentic_swarm_bench.runner.direct import _enforce_exit_code

    run = BenchmarkRun(model="m", endpoint="e", started_at="2026-01-01T00:00:00")
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="fresh",
            context_tokens=6000,
            wall_time_s=0.1,
            requests=[
                RequestMetrics(
                    request_id=0,
                    user_id=0,
                    task_id="P1",
                    context_profile="fresh",
                    context_tokens=6000,
                    error="ConnectionError: refused",
                )
            ],
        )
    )
    with pytest.raises(SystemExit) as exc:
        _enforce_exit_code(run)
    assert exc.value.code == 1


def test_enforce_exit_code_ok_when_any_success():
    from agentic_swarm_bench.runner.direct import _enforce_exit_code

    run = BenchmarkRun(model="m", endpoint="e", started_at="2026-01-01T00:00:00")
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="fresh",
            context_tokens=6000,
            wall_time_s=0.1,
            requests=[
                RequestMetrics(
                    request_id=0,
                    user_id=0,
                    task_id="P1",
                    context_profile="fresh",
                    context_tokens=6000,
                    ttft_ms=10.0,
                    total_time_s=0.1,
                    completion_tokens=1,
                )
            ],
        )
    )
    _enforce_exit_code(run)  # must not raise


# ---------------------------------------------------------------------------
# Bug 9 – --max-tokens CLI caps per-task max_output_tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_max_tokens_caps_per_task_default(monkeypatch):
    from agentic_swarm_bench.runner import direct as runner_module

    seen_max_tokens: list[int] = []

    async def fake_send(**kwargs):
        seen_max_tokens.append(kwargs["max_tokens"])
        on_complete = kwargs.get("on_complete")
        if on_complete:
            on_complete()
        return RequestMetrics(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            task_id=kwargs["task_id"],
            context_profile=kwargs["context_profile"],
            context_tokens=kwargs["context_tokens"],
            ttft_ms=1.0,
            total_time_s=0.01,
            completion_tokens=1,
        )

    monkeypatch.setattr(runner_module, "_send_streaming_request", fake_send)

    config = BenchmarkConfig(endpoint="http://x", model="m", max_output_tokens=32)
    tasks = [{"id": "P1", "prompt": "hi", "max_output_tokens": 8000}]  # CLI must cap this

    await runner_module.run_scenario(
        config=config,
        url="http://x/v1/chat/completions",
        headers={},
        num_users=1,
        context_tokens=6000,
        context_profile="fresh",
        tasks=tasks,
        defeat_cache=False,
    )
    assert seen_max_tokens == [32]


# ---------------------------------------------------------------------------
# Bug 10 – --extra-body and --enable-thinking merge into payload
# ---------------------------------------------------------------------------


def test_speed_extra_body_is_accepted():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "-p", "fresh", "-u", "1",
         "--extra-body", '{"top_p": 0.95}', "--enable-thinking", "--dry-run"],
    )
    assert result.exit_code == 0, result.output


def test_merge_extra_body_combines_json_and_enable_thinking():
    from agentic_swarm_bench.cli import _merge_extra_body

    merged = _merge_extra_body('{"top_p": 0.95}', True)
    assert merged["top_p"] == 0.95
    assert merged["chat_template_kwargs"]["enable_thinking"] is True


def test_merge_extra_body_flag_alone():
    from agentic_swarm_bench.cli import _merge_extra_body

    merged = _merge_extra_body(None, True)
    assert merged["chat_template_kwargs"]["enable_thinking"] is True


def test_merge_extra_body_neither_returns_none():
    from agentic_swarm_bench.cli import _merge_extra_body

    assert _merge_extra_body(None, False) is None


def test_speed_invalid_extra_body_json_errors():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "-p", "fresh",
         "--extra-body", "not-valid-json", "--dry-run"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Bug 11 – JSON output includes top-level verdict + aggregate percentiles
# ---------------------------------------------------------------------------


def test_benchmark_run_to_dict_includes_summary_and_verdict():
    run = BenchmarkRun(model="m", endpoint="e", started_at="2026-01-01T00:00:00")
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="fresh",
            context_tokens=6000,
            wall_time_s=1.0,
            requests=[
                RequestMetrics(
                    request_id=i,
                    user_id=0,
                    task_id="P1",
                    context_profile="fresh",
                    context_tokens=6000,
                    ttft_ms=100.0 + i,
                    total_time_s=1.0,
                    completion_tokens=50,
                    tok_per_sec=60.0,
                    itl_ms=[10.0] * 10,
                )
                for i in range(5)
            ],
        )
    )
    d = run.to_dict()
    assert "verdict" in d
    assert "summary" in d
    summary = d["summary"]
    for key in ("ttft_ms", "tok_per_sec", "itl_ms"):
        assert key in summary
        assert "p50" in summary[key]
        assert "p95" in summary[key]
        assert "p99" in summary[key]


# ---------------------------------------------------------------------------
# Bug 12 – Ctrl+C during speed preserves progress and exits 130
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speed_keyboard_interrupt_saves_partial_and_exits_130(monkeypatch, tmp_path):
    # Completed scenarios must survive a mid-sweep Ctrl+C; exit code 130 lets
    # shell pipelines distinguish a cancel from a failure.
    from agentic_swarm_bench.runner import direct as runner_module

    call_count = {"n": 0}

    async def fake_run_scenario(*, config, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ScenarioResult(
                num_users=kwargs["num_users"],
                context_profile=kwargs["context_profile"],
                context_tokens=kwargs["context_tokens"],
                wall_time_s=0.1,
                requests=[
                    RequestMetrics(
                        request_id=0,
                        user_id=0,
                        task_id="P1",
                        context_profile=kwargs["context_profile"],
                        context_tokens=kwargs["context_tokens"],
                        ttft_ms=100.0,
                        total_time_s=0.1,
                        completion_tokens=1,
                        tok_per_sec=10.0,
                    )
                ],
                cache_mode=kwargs.get("cache_mode_label"),
            )
        raise KeyboardInterrupt

    async def no_models_check(*a, **kw):
        return None

    monkeypatch.setattr(runner_module, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(runner_module, "_check_models_endpoint", no_models_check)

    out = tmp_path / "partial.json"
    config = BenchmarkConfig(
        endpoint="http://x",
        model="m",
        users=1,
        suite="quick",  # expands to several scenarios so the 2nd raises
        output=str(out),
        output_format="json",
        model_context_length=32_000,
    )

    with pytest.raises(SystemExit) as exc:
        await runner_module.run_speed_benchmark(config)
    assert exc.value.code == 130
    assert out.exists()
    saved = json.loads(out.read_text())
    assert len(saved["scenarios"]) == 1
    assert saved["scenarios"][0]["requests"][0]["completion_tokens"] == 1


@pytest.mark.asyncio
async def test_run_scenario_cancel_midflight_preserves_completed_requests():
    # A single user finishing while peers are still streaming must survive the
    # cancel - the partial ScenarioResult is carried out on _ScenarioCancelled.
    from agentic_swarm_bench.runner import direct as runner_module

    async def mixed_send(**kw):
        uid = kw["user_id"]
        if uid == 0:
            return RequestMetrics(
                request_id=kw["request_id"],
                user_id=uid,
                task_id=kw["task_id"],
                context_profile=kw["context_profile"],
                context_tokens=kw["context_tokens"],
                ttft_ms=5.0,
                total_time_s=0.01,
                completion_tokens=1,
                tok_per_sec=100.0,
            )
        await asyncio.sleep(10)
        return RequestMetrics(
            request_id=kw["request_id"],
            user_id=uid,
            task_id=kw["task_id"],
            context_profile=kw["context_profile"],
            context_tokens=kw["context_tokens"],
            ttft_ms=5.0,
        )

    original_send = runner_module._send_streaming_request
    runner_module._send_streaming_request = mixed_send
    try:
        cfg = BenchmarkConfig(endpoint="http://x", model="m", max_output_tokens=16)
        task = asyncio.create_task(
            runner_module.run_scenario(
                config=cfg,
                url="http://x",
                headers={},
                num_users=4,
                context_tokens=6000,
                context_profile="fresh",
                tasks=[{"id": "P1", "prompt": "hi"}],
                repetitions=1,
                request_counter=itertools.count(0),
                defeat_cache=False,
            )
        )
        await asyncio.sleep(0.2)  # user 0 finishes, users 1-3 still in flight
        task.cancel()
        with pytest.raises(runner_module._ScenarioCancelled) as exc_info:
            await task
        scenario = exc_info.value.scenario
        assert len(scenario.requests) == 1, "partial save dropped user-0"
        assert scenario.requests[0].user_id == 0
        assert scenario.requests[0].completion_tokens == 1
    finally:
        runner_module._send_streaming_request = original_send


# ---------------------------------------------------------------------------
# Bug 13 – -u 0 and other nonsense numeric flags rejected by click.IntRange
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["0", "-1", "-5"])
def test_speed_users_must_be_positive(bad_value):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "-p", "fresh",
         "-u", bad_value, "--dry-run"],
    )
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "bad_flag,bad_value",
    [("--context-tokens", "0"), ("--max-tokens", "0"),
     ("--timeout", "0"), ("--repetitions", "0")],
)
def test_speed_numeric_flags_must_be_positive(bad_flag, bad_value):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["speed", "-e", "http://x", "-m", "m", "-p", "fresh",
         "-u", "1", bad_flag, bad_value, "--dry-run"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Bug 14 – cache_mode is a dedicated field on ScenarioResult
# ---------------------------------------------------------------------------


def test_scenario_result_cache_mode_serializes():
    s = ScenarioResult(
        num_users=1,
        context_profile="fresh",
        context_tokens=6000,
        wall_time_s=1.0,
        requests=[],
        cache_mode="allcold",
    )
    assert s.to_dict()["cache_mode"] == "allcold"


# ---------------------------------------------------------------------------
# Bug 15 – cache-mode realistic produces a per-pass verdict block
# ---------------------------------------------------------------------------


def test_realistic_report_has_per_pass_verdict():
    from agentic_swarm_bench.report.markdown import generate_report

    run = BenchmarkRun(model="m", endpoint="e", started_at="2026-01-01T00:00:00")
    for label in ("allcold", "allwarm"):
        run.scenarios.append(
            ScenarioResult(
                num_users=2,
                context_profile=f"fresh ({label})",
                context_tokens=6000,
                wall_time_s=1.0,
                cache_mode=label,
                requests=[
                    RequestMetrics(
                        request_id=i,
                        user_id=0,
                        task_id="P1",
                        context_profile=f"fresh ({label})",
                        context_tokens=6000,
                        ttft_ms=200.0 if label == "allcold" else 50.0,
                        total_time_s=1.0,
                        completion_tokens=50,
                        tok_per_sec=60.0,
                    )
                    for i in range(3)
                ],
            )
        )
    report = generate_report(run)
    assert "Per-pass verdict" in report
    assert "allcold" in report
    assert "allwarm" in report


# ---------------------------------------------------------------------------
# Bug 16 – error summary groups by error type (ConnectionError, HTTP 401, ...)
# ---------------------------------------------------------------------------


def test_error_summary_groups_by_type(capsys):
    from agentic_swarm_bench.runner.direct import _print_error_summary

    run = BenchmarkRun(model="m", endpoint="e", started_at="t")
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile="fresh",
            context_tokens=6000,
            wall_time_s=0.1,
            requests=[
                RequestMetrics(
                    request_id=i,
                    user_id=0,
                    task_id="P1",
                    context_profile="fresh",
                    context_tokens=6000,
                    error="ConnectionError (api.swarmone.ai after 5.0s): refused",
                )
                for i in range(3)
            ],
        )
    )
    _print_error_summary(run)
    out = capsys.readouterr().out
    assert "ConnectionError" in out
    assert "× 3" in out


# ---------------------------------------------------------------------------
# Bug 17 – request_id is globally monotonic across scenarios / repetitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_is_monotonic_across_scenarios(monkeypatch):
    from agentic_swarm_bench.runner import direct as runner_module

    all_ids: list[int] = []

    async def fake_send(**kwargs):
        all_ids.append(kwargs["request_id"])
        on_complete = kwargs.get("on_complete")
        if on_complete:
            on_complete()
        return RequestMetrics(
            request_id=kwargs["request_id"],
            user_id=kwargs["user_id"],
            task_id=kwargs["task_id"],
            context_profile=kwargs["context_profile"],
            context_tokens=kwargs["context_tokens"],
            ttft_ms=1.0,
            total_time_s=0.01,
            completion_tokens=1,
        )

    monkeypatch.setattr(runner_module, "_send_streaming_request", fake_send)

    config = BenchmarkConfig(endpoint="http://x", model="m", max_output_tokens=16)
    tasks = [{"id": "P1", "prompt": "hi"}]
    counter = itertools.count(0)

    for _ in range(3):
        await runner_module.run_scenario(
            config=config,
            url="http://x/v1/chat/completions",
            headers={},
            num_users=2,
            context_tokens=6000,
            context_profile="fresh",
            tasks=tasks,
            request_counter=counter,
            defeat_cache=False,
        )

    assert all_ids == list(range(6))


# ---------------------------------------------------------------------------
# Bug 18 – asb compare distinguishes no-overlap from all-failure
# ---------------------------------------------------------------------------


def _single_scenario_run(model: str, profile: str) -> BenchmarkRun:
    run = BenchmarkRun(model=model, endpoint="e", started_at="t")
    run.scenarios.append(
        ScenarioResult(
            num_users=1,
            context_profile=profile,
            context_tokens=6000,
            wall_time_s=1.0,
            requests=[
                RequestMetrics(
                    request_id=0,
                    user_id=0,
                    task_id="P1",
                    context_profile=profile,
                    context_tokens=6000,
                    ttft_ms=100.0,
                    total_time_s=1.0,
                    completion_tokens=50,
                    tok_per_sec=60.0,
                )
            ],
        )
    )
    return run


def test_compare_no_overlap_says_so_explicitly():
    a = _single_scenario_run("a", "fresh")
    b = _single_scenario_run("b", "short")
    report = generate_comparison(a, b)
    assert "No shared" in report
    # And must not falsely blame failures - the sides both succeeded.
    assert "all requests failed" not in report
    assert "all overlapping scenarios had zero completions" not in report


# ---------------------------------------------------------------------------
# Bug 19 – --help Modes list includes every subcommand
# ---------------------------------------------------------------------------


def test_help_modes_list_is_complete():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    # Only check the docstring area, before the auto-generated "Commands:" section.
    head = result.output.split("Commands:")[0]
    for mode in ("speed", "record", "replay", "list-scenarios", "report", "compare"):
        assert mode in head, f"{mode} missing from --help 'Modes' list"


# ---------------------------------------------------------------------------
# Bug 20 – ITL sub-millisecond renders as <1ms (not 0ms) and zero as dash
# ---------------------------------------------------------------------------


def test_fmt_ms_sub_millisecond_renders_as_less_than_one():
    assert _fmt_ms(0.4) == "<1ms"
    assert _fmt_ms(0.01) == "<1ms"


def test_fmt_ms_zero_renders_as_dash():
    assert _fmt_ms(0) == "-"


def test_fmt_ms_integer_milliseconds_still_rounded():
    assert _fmt_ms(42) == "42ms"
    assert _fmt_ms(999.4) == "999ms"

