"""Direct endpoint benchmark runner.

Sends streaming requests directly to any OpenAI-compatible endpoint,
measuring TTFT, tok/s, ITL, and aggregate throughput at various
concurrency levels and context sizes.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from agentic_swarm_bench.config import BenchmarkConfig, resolve_endpoint
from agentic_swarm_bench.metrics.collector import (
    BenchmarkRun,
    RequestMetrics,
    ScenarioResult,
    is_context_length_error,
)
from agentic_swarm_bench.metrics.stats import ScenarioStats, analyze_scenario
from agentic_swarm_bench.proxy.padding import poison_messages
from agentic_swarm_bench.tasks.context.codebase_context import build_messages
from agentic_swarm_bench.tasks.registry import get_tasks

console = Console()

ACCENT = "cyan"
DIM = "dim"
OK_COLOR = "green"
WARN_COLOR = "yellow"
ERR_COLOR = "red"


def _fmt_ms(ms: float) -> str:
    """Format milliseconds into a human-friendly duration string."""
    if ms <= 0:
        return "-"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 10_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 1000:.0f}s"


def _build_headers(config: BenchmarkConfig) -> dict:
    """Build request headers with the configured auth method."""
    headers = {"Content-Type": "application/json"}
    if not config.api_key:
        return headers

    if config.api_key_header.lower() == "authorization":
        headers["Authorization"] = f"Bearer {config.api_key}"
    else:
        headers[config.api_key_header] = config.api_key
    return headers


async def _send_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    messages: list[dict],
    max_tokens: int,
    user_id: int,
    task_id: str,
    context_profile: str,
    context_tokens: int,
    timeout: float,
    on_complete: object = None,
    on_token: object = None,
) -> RequestMetrics:
    """Send a single streaming request and collect timing metrics."""
    metrics = RequestMetrics(
        user_id=user_id,
        task_id=task_id,
        context_profile=context_profile,
        context_tokens=context_tokens,
    )

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    first_visible_time = None
    first_thinking_time = None
    last_token_time = start
    token_count = 0
    thinking_count = 0

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                metrics.error = f"HTTP {resp.status_code}: {body.decode()[:500]}"
                if on_complete:
                    on_complete()
                return metrics

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)

                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    now = time.perf_counter()

                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")

                    if reasoning:
                        thinking_count += 1
                        if first_thinking_time is None:
                            first_thinking_time = now
                            metrics.ttft_thinking_ms = (now - start) * 1000

                    if content and first_visible_time is None:
                        first_visible_time = now
                        metrics.ttft_visible_ms = (now - start) * 1000

                    if not reasoning and not content:
                        continue

                    if first_token_time is None:
                        first_token_time = now
                        metrics.ttft_ms = (now - start) * 1000
                    else:
                        itl = (now - last_token_time) * 1000
                        metrics.itl_ms.append(itl)
                    last_token_time = now
                    token_count += 1

                    if on_token:
                        on_token(user_id, token_count, (now - start))

    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)[:500]}"
        if on_complete:
            on_complete()
        return metrics

    end = time.perf_counter()
    metrics.total_time_s = end - start
    metrics.completion_tokens = token_count
    metrics.thinking_tokens = thinking_count

    if token_count > 0 and first_token_time is not None:
        metrics.decode_time_s = end - first_token_time
        if metrics.decode_time_s > 0:
            metrics.tok_per_sec = token_count / metrics.decode_time_s

    if metrics.ttft_ms > 0 and metrics.prompt_tokens > 0:
        prefill_s = metrics.ttft_ms / 1000
        metrics.prefill_tok_per_sec = metrics.prompt_tokens / prefill_s
    elif metrics.ttft_ms > 0 and context_tokens > 0:
        prefill_s = metrics.ttft_ms / 1000
        metrics.prefill_tok_per_sec = context_tokens / prefill_s

    if on_complete:
        on_complete()
    return metrics


async def run_scenario(
    config: BenchmarkConfig,
    url: str,
    headers: dict,
    num_users: int,
    context_tokens: int,
    context_profile: str,
    tasks: list[dict],
    progress: Progress | None = None,
    scenario_idx: int = 0,
    total_scenarios: int = 0,
    defeat_cache: bool = True,
) -> ScenarioResult:
    """Run one benchmark scenario: N concurrent users at a given context size."""
    ctx_k = context_tokens // 1000
    user_word = "user" if num_users == 1 else "users"
    short_label = f"{context_profile} · {ctx_k}K · {num_users} {user_word}"

    if total_scenarios:
        counter = f"[{DIM}]{scenario_idx}/{total_scenarios}[/{DIM}]"
        console.print(f"\n  {counter} [bold]─[/bold] {short_label}")
    else:
        console.print(f"\n  [bold]─[/bold] {short_label}")

    progress_label = f"{context_profile} · {ctx_k}K · {num_users}{user_word[0]}"
    task_id = None
    if progress:
        task_id = progress.add_task(f"    {progress_label}", total=num_users)

    token_counts: dict[int, int] = {}
    completed_count = 0

    def make_complete_callback():
        nonlocal completed_count

        def cb():
            nonlocal completed_count
            completed_count += 1
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    completed=completed_count,
                    description=_live_description(
                        progress_label,
                        token_counts,
                    ),
                )

        return cb

    last_progress_update = 0.0

    def make_token_callback():
        nonlocal last_progress_update

        def cb(user_id: int, tokens: int, elapsed: float):
            nonlocal last_progress_update
            token_counts[user_id] = tokens
            now = time.monotonic()
            if progress and task_id is not None and now - last_progress_update > 0.25:
                last_progress_update = now
                progress.update(
                    task_id,
                    description=_live_description(
                        progress_label,
                        token_counts,
                    ),
                )

        return cb

    coros_args = []
    for i in range(num_users):
        task = tasks[i % len(tasks)]
        prompt = task["prompt"]
        tid = task["id"]
        max_tok = task.get("max_output_tokens", config.max_output_tokens)
        msgs = build_messages(
            prompt,
            context_tokens,
            random_seed=None if not config.random_context else (i + time.time_ns()),
        )
        if defeat_cache:
            msgs = poison_messages(msgs, seed=f"speed-{tid}-u{i}-{time.time_ns()}")
        coros_args.append((msgs, max_tok, tid))

    async with httpx.AsyncClient() as client:
        coros = [
            _send_streaming_request(
                client=client,
                url=url,
                model=config.model,
                headers=headers,
                messages=msgs,
                max_tokens=max_tok,
                user_id=i,
                task_id=tid,
                context_profile=context_profile,
                context_tokens=context_tokens,
                timeout=config.timeout,
                on_complete=make_complete_callback(),
                on_token=make_token_callback(),
            )
            for i, (msgs, max_tok, tid) in enumerate(coros_args)
        ]

        wall_start = time.perf_counter()
        results = await asyncio.gather(*coros)
        wall_time = time.perf_counter() - wall_start

    if progress and task_id is not None:
        progress.remove_task(task_id)

    scenario = ScenarioResult(
        num_users=num_users,
        context_profile=context_profile,
        context_tokens=context_tokens,
        wall_time_s=wall_time,
        requests=list(results),
    )

    stats = analyze_scenario(scenario)
    _print_scenario_stats(stats, scenario, verbose=config.verbose)
    return scenario


def _live_description(
    label: str,
    token_counts: dict[int, int],
) -> str:
    """Build a live-updating description string for the progress bar."""
    total_tokens = sum(token_counts.values())
    if total_tokens == 0:
        return f"    {label}  waiting for first token…"
    return f"    {label}  {total_tokens:,} tok"


def _print_scenario_stats(
    stats: ScenarioStats,
    scenario: ScenarioResult | None = None,
    verbose: bool = False,
) -> None:
    """Print structured metrics for one completed scenario."""
    if stats.successful == 0:
        fail = f"{stats.total_requests}/{stats.total_requests}"
        console.print(f"    [{ERR_COLOR}]✗ {fail} failed[/{ERR_COLOR}]")
        if scenario:
            _print_failure_details(scenario.failures)
        return

    ok_str = f"[{OK_COLOR}]✓ {stats.successful}/{stats.total_requests}[/{OK_COLOR}]"

    tok_s = f"[bold {ACCENT}]{stats.tok_per_sec.median:.1f}[/bold {ACCENT}] tok/s"

    ttft_p50 = _fmt_ms(stats.ttft_ms.median)
    ttft_p99 = _fmt_ms(stats.ttft_ms.p99)
    if ttft_p50 == ttft_p99:
        ttft_str = f"TTFT {ttft_p50}"
    else:
        ttft_str = f"TTFT {ttft_p50} [{DIM}]p99 {ttft_p99}[/{DIM}]"

    itl_str = f"ITL {_fmt_ms(stats.itl_ms.median)}"

    agg_str = f"[{DIM}]Σ {stats.aggregate_tok_per_sec:.0f} tok/s[/{DIM}]"

    console.print(f"    {ok_str}   {tok_s}   {ttft_str}   {itl_str}   {agg_str}")

    if stats.has_thinking:
        overhead = stats.thinking_overhead_ms
        console.print(
            f"    [{WARN_COLOR}]     thinking overhead "
            f"{_fmt_ms(overhead.median)} p50[/{WARN_COLOR}]"
        )

    if stats.failed > 0 and scenario:
        _print_failure_details(scenario.failures)

    if verbose and scenario:
        _print_verbose_requests(scenario)


def _print_failure_details(failures: list) -> None:
    """Print why requests failed, grouped by reason."""
    empty_response = [r for r in failures if r.error is None]
    with_error = [r for r in failures if r.error is not None]
    ctx_len = [r for r in with_error if is_context_length_error(r.error)]
    other_error = [r for r in with_error if not is_context_length_error(r.error)]

    if empty_response:
        console.print(
            f"    [{WARN_COLOR}]{len(empty_response)} empty response "
            f"- check model name, max_tokens, or content policy[/{WARN_COLOR}]"
        )
    if ctx_len:
        console.print(
            f"    [{WARN_COLOR}]{len(ctx_len)} request(s) exceeded the model's"
            f" context window - use --model-context-length to skip them[/{WARN_COLOR}]"
        )
    for r in other_error:
        console.print(f"    [{ERR_COLOR}]✗ {r.task_id}: {r.error[:120]}[/{ERR_COLOR}]")


def _print_verbose_requests(scenario: ScenarioResult) -> None:
    """Print per-request detail table for verbose mode."""
    for r in scenario.requests:
        if r.error:
            console.print(
                f"         [{ERR_COLOR}]✗ u{r.user_id} {r.task_id}  {r.error[:80]}[/{ERR_COLOR}]"
            )
            continue

        ttft = _fmt_ms(r.ttft_ms)
        tok_s = f"{r.tok_per_sec:.1f}" if r.tok_per_sec > 0 else "-"
        console.print(
            f"         [{DIM}]u{r.user_id} {r.task_id}  "
            f"{r.completion_tokens} tok  {r.total_time_s:.1f}s  "
            f"TTFT {ttft}  {tok_s} tok/s[/{DIM}]"
        )


async def _check_models_endpoint(endpoint: str, headers: dict) -> None:
    """Try to list models from the endpoint; log if unavailable."""
    base = endpoint.rstrip("/")
    if "/v1/chat/completions" in base:
        base = base.rsplit("/v1/chat/completions", 1)[0]
    models_url = base.rstrip("/") + "/v1/models"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(models_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                model_ids = [m.get("id", "?") for m in data.get("data", [])]
                if model_ids:
                    console.print(
                        f"  Available models: {', '.join(model_ids[:10])}"
                        f"{'...' if len(model_ids) > 10 else ''}"
                    )
                    return
            pass  # model listing unavailable - not required, skip silently
    except Exception:
        pass  # model listing unavailable - not required, skip silently


async def run_speed_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Run the full speed benchmark across all configured scenarios."""
    tasks = get_tasks(task_range=config.task_range, tier=config.tier, tags=config.tags)
    if not tasks:
        tasks = get_tasks(task_range="p1-p25")

    url = resolve_endpoint(config.endpoint)
    headers = _build_headers(config)

    run = BenchmarkRun(
        model=config.model,
        endpoint=config.endpoint,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    scenarios = config.resolved_scenarios
    if not scenarios:
        console.print(
            f"[bold red]Error:[/] No scenarios resolved for suite={config.suite!r} "
            f"with model-context-length={config.model_context_length}.\n"
            "Try a larger --model-context-length or a different --suite.",
        )
        raise SystemExit(1)
    cache_mode = getattr(config, "cache_mode", "cold")

    max_context_tokens = max((t for _, _, t in scenarios), default=0)
    profiles = sorted(
        set(s[1] for s in scenarios),
        key=lambda p: next((t for _, pn, t in scenarios if pn == p), 0),
    )
    user_counts = sorted(set(s[0] for s in scenarios))

    parsed = urlparse(config.endpoint)
    short_host = parsed.hostname or config.endpoint

    profile_range = f"{profiles[0]}–{profiles[-1]}" if len(profiles) > 1 else profiles[0]

    header_lines = Text()
    header_lines.append("Model     ", style=DIM)
    header_lines.append(f"{config.model}\n", style="bold")
    header_lines.append("Endpoint  ", style=DIM)
    header_lines.append(f"{short_host}\n")
    header_lines.append("Scenarios ", style=DIM)
    header_lines.append(
        f"{len(scenarios)}  "
        f"({profile_range} × {', '.join(str(u) for u in user_counts)} users)  "
        f"· {cache_mode} cache\n"
    )
    header_lines.append("Tasks     ", style=DIM)
    header_lines.append(f"{len(tasks)} loaded  · max {max_context_tokens // 1000}K context")

    if config.model_context_length is not None:
        all_profile_tokens = config._resolve_profile_tokens()
        skipped = [p for p, t in all_profile_tokens if t > config.model_context_length]
        header_lines.append(f"  · {config.model_context_length // 1000}K model window")
        if skipped:
            header_lines.append(
                f"\n[{WARN_COLOR}]Skipping profiles exceeding window: "
                f"{', '.join(skipped)}[/{WARN_COLOR}]"
            )

    console.print()
    console.print(
        Panel(
            header_lines,
            title="[bold]asb speed[/bold]",
            title_align="left",
            border_style=DIM,
            padding=(0, 2),
        )
    )

    await _check_models_endpoint(config.endpoint, headers)

    if config.dry_run:
        _print_dry_run(config, tasks, scenarios, url)
        return run

    passes = _get_cache_passes(cache_mode)
    total_scenario_count = len(scenarios) * len(passes)

    scenario_num = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for pass_name, defeat in passes:
            if len(passes) > 1:
                console.print(f"\n[bold magenta]Pass: {pass_name}[/bold magenta]")

            for num_users, profile, tokens in scenarios:
                scenario_num += 1
                scenario = await run_scenario(
                    config=config,
                    url=url,
                    headers=headers,
                    num_users=num_users,
                    context_tokens=tokens,
                    context_profile=(f"{profile} ({pass_name})" if len(passes) > 1 else profile),
                    tasks=tasks,
                    progress=progress,
                    scenario_idx=scenario_num,
                    total_scenarios=total_scenario_count,
                    defeat_cache=defeat,
                )
                run.scenarios.append(scenario)
                await asyncio.sleep(2)

    console.print()
    _print_summary_table(run)

    if config.output:
        _save_outputs(config, run)

    return run


def _print_dry_run(
    config: BenchmarkConfig,
    tasks: list[dict],
    scenarios: list[tuple[int, str, int]],
    url: str,
) -> None:
    """Print what would be run without actually sending requests."""
    msg = f"[{WARN_COLOR} bold]DRY RUN - no requests will be sent[/{WARN_COLOR} bold]"
    console.print(f"\n  {msg}\n")

    cache_label = {"cold": "poisoned (cold)", "warm": "allowed (warm)", "both": "cold + warm"}
    rows = [
        ("URL", url),
        ("Auth", config.api_key_header),
        ("Max output", f"{config.max_output_tokens} tokens"),
        ("Cache", cache_label.get(getattr(config, "cache_mode", "cold"), "poisoned")),
        ("Timeout", f"{config.timeout:.0f}s"),
    ]
    if config.random_context:
        rows.append(("Context", "randomized"))
    if config.model_context_length is not None:
        rows.append(("Model window", f"{config.model_context_length // 1000}K tokens"))

    for label, value in rows:
        console.print(f"  [{DIM}]{label:<12}[/{DIM}] {value}")

    console.print(f"\n  [{DIM}]Tasks ({len(tasks)}):[/{DIM}]")
    for t in tasks[:5]:
        console.print(f"    [{DIM}]{t['id']}[/{DIM}]  {t['prompt'][:55]}…")
    if len(tasks) > 5:
        console.print(f"    [{DIM}]…and {len(tasks) - 5} more[/{DIM}]")

    max_context_tokens = max((t for _, _, t in scenarios), default=0)
    max_k = max_context_tokens // 1000
    console.print(f"\n  [{DIM}]Scenarios ({len(scenarios)}, max {max_k}K):[/{DIM}]")
    for users, profile, tokens in scenarios:
        user_word = "user" if users == 1 else "users"
        console.print(f"    {profile} · {tokens // 1000}K · {users} {user_word}")

    sample_msgs = build_messages(tasks[0]["prompt"], 6000)
    total_chars = sum(len(m["content"]) for m in sample_msgs)
    n_msgs = len(sample_msgs)
    console.print(
        f"\n  [{DIM}]Sample request (P1 at 6K): ~{total_chars:,} chars, {n_msgs} msgs[/{DIM}]"
    )


def _save_outputs(config: BenchmarkConfig, run: BenchmarkRun) -> None:
    """Save results in the requested format(s), auto-creating directories."""
    output = config.output
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.output_format == "json":
        json_path = output if output.endswith(".json") else output + ".json"
        run.save(json_path)
        console.print(f"\n  [{DIM}]Saved →[/{DIM}]  {json_path}")
        return

    json_path = _derive_json_path(output)
    run.save(json_path)
    console.print(f"\n  [{DIM}]Saved →[/{DIM}]  {json_path}")

    if output.endswith(".md"):
        from agentic_swarm_bench.report.markdown import generate_report

        report = generate_report(run, json_path=json_path)
        with open(output, "w") as f:
            f.write(report)
        console.print(f"  [{DIM}]Saved →[/{DIM}]  {output}")


def _get_cache_passes(cache_mode: str) -> list[tuple[str, bool]]:
    """Return (label, defeat_cache) pairs for the selected cache mode."""
    if cache_mode == "both":
        return [("cold", True), ("warm", False)]
    if cache_mode == "warm":
        return [("warm", False)]
    return [("cold", True)]



def _derive_json_path(output: str) -> str:
    """Convert output path to .json if it ends with .md."""
    if output.endswith(".md"):
        return output[:-3] + ".json"
    if output.endswith(".json"):
        return output
    return output + ".json"


def _print_summary_table(run: BenchmarkRun) -> None:
    """Print a final summary table across all scenarios with verdict."""
    from agentic_swarm_bench.report.markdown import (
        _verdict_for_stats,
        _verdict_label,
    )

    table = Table(
        title=f"[bold]{run.model}[/bold]",
        title_style="",
        border_style=DIM,
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Users", justify="right", style=DIM)
    table.add_column("Context", justify="left")
    table.add_column("Tok/s", justify="right", style=f"bold {ACCENT}")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right", style=DIM)
    table.add_column("ITL", justify="right")
    table.add_column("Σ tok/s", justify="right", style=DIM)
    table.add_column("OK", justify="right")

    verdict_stats = None

    for scenario in run.scenarios:
        stats = analyze_scenario(scenario)
        if stats.successful == 0:
            table.add_row(
                f"[{ERR_COLOR}]●[/{ERR_COLOR}]",
                str(stats.num_users),
                stats.context_profile,
                f"[{ERR_COLOR}]FAIL[/{ERR_COLOR}]",
                "-",
                "-",
                "-",
                "-",
                f"0/{stats.total_requests}",
            )
            continue

        verdict = _verdict_for_stats(stats)
        color_map = {"good": OK_COLOR, "ok": WARN_COLOR, "poor": ERR_COLOR}
        dot_color = color_map.get(verdict, "white")

        profile_base = stats.context_profile.split("(")[0].strip()
        if profile_base in ("medium", "long") and (
            verdict_stats is None or stats.num_users < verdict_stats.num_users
        ):
            verdict_stats = stats

        ok_str = (
            f"[{OK_COLOR}]{stats.successful}[/{OK_COLOR}]/{stats.total_requests}"
            if stats.successful == stats.total_requests
            else f"{stats.successful}/{stats.total_requests}"
        )

        table.add_row(
            f"[{dot_color}]●[/{dot_color}]",
            str(stats.num_users),
            stats.context_profile,
            f"{stats.tok_per_sec.median:.1f}",
            _fmt_ms(stats.ttft_ms.median),
            _fmt_ms(stats.ttft_ms.p99),
            _fmt_ms(stats.itl_ms.median),
            f"{stats.aggregate_tok_per_sec:.0f}",
            ok_str,
        )

    console.print(table)

    if verdict_stats is None:
        successful = [analyze_scenario(s) for s in run.scenarios if len(s.successes) > 0]
        if successful:
            verdict_stats = successful[0]

    if verdict_stats:
        verdict = _verdict_for_stats(verdict_stats)
        label = _verdict_label(verdict)
        color_map = {"good": OK_COLOR, "ok": WARN_COLOR, "poor": ERR_COLOR}
        color = color_map.get(verdict, "white")
        console.print(
            f"\n  [{color} bold]{label}[/{color} bold]  "
            f"[{DIM}]at {verdict_stats.context_profile} context[/{DIM}]"
        )

    ctx_len_failures = sum(
        1 for s in run.scenarios for r in s.failures if is_context_length_error(r.error)
    )
    if ctx_len_failures:
        console.print(
            f"\n  [{WARN_COLOR}]⚠  {ctx_len_failures} request(s) failed because the"
            f" prompt exceeded the model's context window.[/{WARN_COLOR}]"
        )
        console.print(
            f"  [{WARN_COLOR}]   Use [bold]--model-context-length N[/bold]"
            f" to skip scenarios that exceed N tokens.[/{WARN_COLOR}]"
        )
