"""Replay recorded workloads against any OpenAI-compatible endpoint.

Takes a JSONL workload (from `acb record` or built-in) and replays each
request against a target endpoint, collecting the same metrics as
`acb speed`. This lets you compare how a real agentic session performs
across different endpoints, hardware, or configurations.

With ``--users N``, N copies of the workload run concurrently - each
user replays the full session sequentially, but the N sessions overlap,
stressing the endpoint under realistic multi-tenant load.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from agentic_coding_bench.config import BenchmarkConfig, resolve_endpoint
from agentic_coding_bench.metrics.collector import (
    BenchmarkRun,
    RequestMetrics,
    ScenarioResult,
    is_context_length_error,
)
from agentic_coding_bench.metrics.stats import analyze_scenario
from agentic_coding_bench.workloads.registry import Workload, get_workload

console = Console()


def _build_headers(config: BenchmarkConfig) -> dict:
    headers = {"Content-Type": "application/json"}
    if not config.api_key:
        return headers
    if config.api_key_header.lower() == "authorization":
        headers["Authorization"] = f"Bearer {config.api_key}"
    else:
        headers[config.api_key_header] = config.api_key
    return headers


async def _replay_one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    messages: list[dict],
    max_tokens: int,
    seq: int,
    timeout: float,
    user_id: int = 0,
    on_complete=None,
) -> RequestMetrics:
    """Replay a single recorded request and collect timing metrics."""
    metrics = RequestMetrics(
        request_id=seq,
        user_id=user_id,
        task_id=f"replay-{seq}",
        context_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
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
    first_thinking_time = None
    first_visible_time = None
    last_token_time = start
    token_count = 0

    try:
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
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
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    if not content and not reasoning:
                        continue

                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        metrics.ttft_ms = (now - start) * 1000
                    else:
                        metrics.itl_ms.append((now - last_token_time) * 1000)
                    last_token_time = now
                    token_count += 1

                    if reasoning:
                        metrics.thinking_tokens += 1
                        if first_thinking_time is None:
                            first_thinking_time = now
                            metrics.ttft_thinking_ms = (now - start) * 1000
                    if content and first_visible_time is None:
                        first_visible_time = now
                        metrics.ttft_visible_ms = (now - start) * 1000

    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)[:500]}"
        if on_complete:
            on_complete()
        return metrics

    end = time.perf_counter()
    metrics.total_time_s = end - start
    metrics.completion_tokens = token_count

    if token_count > 0 and first_token_time is not None:
        metrics.decode_time_s = end - first_token_time
        if metrics.decode_time_s > 0:
            metrics.tok_per_sec = token_count / metrics.decode_time_s

    if metrics.ttft_ms > 0 and metrics.prompt_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.prompt_tokens / (metrics.ttft_ms / 1000)
    elif metrics.ttft_ms > 0 and metrics.context_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.context_tokens / (metrics.ttft_ms / 1000)

    if on_complete:
        on_complete()
    return metrics


def _bucket_label(tokens: int) -> str:
    """Map approximate token count to a context profile label."""
    if tokens < 10_000:
        return "fresh"
    if tokens < 30_000:
        return "short"
    if tokens < 55_000:
        return "medium"
    if tokens < 85_000:
        return "long"
    if tokens < 150_000:
        return "full"
    if tokens < 300_000:
        return "xl"
    return "xxl"


def _slice_entries(
    entries: list,
    slice_tokens: int | None,
) -> list:
    """Take entries in order until cumulative prompt tokens exceed the budget.

    Uses the recorded ``prompt_tokens`` field when available, falling back to
    a char/4 estimate.  Returns a (possibly shorter) list preserving order.
    """
    if slice_tokens is None:
        return entries

    kept: list = []
    cumulative = 0
    for entry in entries:
        tok = entry.prompt_tokens if getattr(entry, "prompt_tokens", None) else (
            sum(len(m.get("content", "")) for m in entry.messages) // 4
        )
        if cumulative + tok > slice_tokens and kept:
            break
        cumulative += tok
        kept.append(entry)
    return kept


def _compute_bucket_wall_time(requests: list[RequestMetrics], num_users: int) -> float:
    """Compute wall time for a bucket of requests.

    Requests are sequential per user but parallel across users, so bucket
    wall time = max across users of (sum of total_time_s per user).
    """
    by_user: dict[int, float] = {}
    for r in requests:
        by_user[r.user_id] = by_user.get(r.user_id, 0.0) + r.total_time_s
    if not by_user:
        return 0.0
    return max(by_user.values())


async def _replay_user_session(
    client: httpx.AsyncClient,
    url: str,
    model_override: str,
    headers: dict,
    entries: list,
    timeout: float,
    user_id: int,
    on_complete=None,
    model_context_length: int | None = None,
) -> list[RequestMetrics]:
    """Replay one full user session sequentially, returning all metrics."""
    results = []
    for entry in entries:
        tokens = sum(len(m.get("content", "")) for m in entry.messages) // 4

        if model_context_length is not None and tokens > model_context_length:
            if on_complete:
                on_complete()
            continue

        model = model_override or entry.model
        m = await _replay_one_request(
            client=client,
            url=url,
            model=model,
            headers=headers,
            messages=entry.messages,
            max_tokens=entry.max_tokens,
            seq=entry.seq,
            timeout=timeout,
            user_id=user_id,
            on_complete=on_complete,
        )
        m.context_profile = _bucket_label(tokens)
        m.context_tokens = tokens
        results.append(m)
    return results


async def replay_workload(
    config: BenchmarkConfig,
    workload_path: str,
    *,
    slice_tokens: int | None = None,
    num_users: int = 1,
    model_context_length: int | None = None,
) -> BenchmarkRun:
    """Replay a recorded workload against the configured endpoint."""
    workload = get_workload(workload_path)
    original_count = workload.total_requests

    if slice_tokens is not None:
        workload.entries = _slice_entries(workload.entries, slice_tokens)

    url = resolve_endpoint(config.endpoint)
    headers = _build_headers(config)

    console.print("\n[bold]AgenticCodingBench -- replay[/bold]")
    console.print(f"  Endpoint: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Workload: {workload.name} ({workload.total_requests} requests)")
    if num_users > 1:
        total = workload.total_requests * num_users
        console.print(f"  Users: {num_users} concurrent ({total} total requests)")
    if slice_tokens is not None:
        console.print(
            f"  Sliced: {workload.total_requests}/{original_count}"
            f" requests (≤{slice_tokens:,} prompt tokens)"
        )
    if model_context_length is not None:
        console.print(f"  Model context length: {model_context_length:,} tokens")
    console.print(f"  Approx tokens: {workload.total_tokens_approx:,} per user")
    console.rule()

    if config.dry_run:
        _print_replay_dry_run(workload, url, num_users)
        return BenchmarkRun(model=config.model, endpoint=config.endpoint)

    run = BenchmarkRun(
        model=config.model,
        endpoint=config.endpoint,
        defeat_cache=False,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    if num_users <= 1:
        await _replay_single_user(
            config, workload, url, headers, run,
            model_context_length=model_context_length,
        )
    else:
        await _replay_multi_user(
            config, workload, url, headers, run, num_users,
            model_context_length=model_context_length,
        )

    console.rule()
    _print_replay_summary(run, workload, num_users)

    if config.output:
        _save_replay_output(config, run)

    return run


async def _replay_single_user(
    config: BenchmarkConfig,
    workload: Workload,
    url: str,
    headers: dict,
    run: BenchmarkRun,
    *,
    model_context_length: int | None = None,
) -> None:
    """Sequential replay: one user, in original workload order."""
    total = len(workload.entries)
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"Replaying ({total} reqs)", total=total)
        all_results: list[tuple[RequestMetrics, str, int]] = []

        async with httpx.AsyncClient() as client:
            for entry in workload.entries:
                tokens = sum(len(m.get("content", "")) for m in entry.messages) // 4
                label = _bucket_label(tokens)

                if model_context_length is not None and tokens > model_context_length:
                    skipped += 1
                    progress.advance(task_id)
                    continue

                model = config.model or entry.model
                m = await _replay_one_request(
                    client=client,
                    url=url,
                    model=model,
                    headers=headers,
                    messages=entry.messages,
                    max_tokens=entry.max_tokens,
                    seq=entry.seq,
                    timeout=config.timeout,
                    on_complete=lambda: progress.advance(task_id),
                )
                m.context_profile = label
                m.context_tokens = tokens
                all_results.append((m, label, tokens))

        progress.remove_task(task_id)

    if skipped:
        console.print(
            f"\n  [dim]Skipped {skipped} request(s) exceeding"
            f" model context length ({model_context_length:,} tokens)[/dim]"
        )

    buckets: dict[str, list[tuple[RequestMetrics, int]]] = {}
    for m, label, tokens in all_results:
        if label not in buckets:
            buckets[label] = []
        buckets[label].append((m, tokens))

    for bucket_label, entries in buckets.items():
        results = [m for m, _ in entries]
        scenario = ScenarioResult(
            num_users=1,
            context_profile=bucket_label,
            context_tokens=max((t for _, t in entries), default=0),
            wall_time_s=_compute_bucket_wall_time(results, num_users=1),
            requests=results,
        )
        run.scenarios.append(scenario)

        stats = analyze_scenario(scenario)
        _print_bucket_stats(bucket_label, stats, num_users=1, scenario=scenario)


async def _replay_multi_user(
    config: BenchmarkConfig,
    workload: Workload,
    url: str,
    headers: dict,
    run: BenchmarkRun,
    num_users: int,
    *,
    model_context_length: int | None = None,
) -> None:
    """Concurrent replay: N users each replay the full session in parallel."""
    total_requests = workload.total_requests * num_users

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"Replaying {num_users} users × {workload.total_requests} reqs",
            total=total_requests,
        )

        async with httpx.AsyncClient() as client:
            coros = [
                _replay_user_session(
                    client=client,
                    url=url,
                    model_override=config.model,
                    headers=headers,
                    entries=workload.entries,
                    timeout=config.timeout,
                    user_id=uid,
                    on_complete=lambda: progress.advance(task_id),
                    model_context_length=model_context_length,
                )
                for uid in range(num_users)
            ]
            all_results = await asyncio.gather(*coros)

        progress.remove_task(task_id)

    flat_results: list[RequestMetrics] = []
    for user_results in all_results:
        flat_results.extend(user_results)

    if model_context_length is not None:
        total_entries = len(workload.entries) * num_users
        skipped = total_entries - len(flat_results)
        if skipped > 0:
            console.print(
                f"\n  [dim]Skipped {skipped} request(s) exceeding"
                f" model context length ({model_context_length:,} tokens)[/dim]"
            )

    buckets: dict[str, list[RequestMetrics]] = {}
    for m in flat_results:
        if m.context_profile not in buckets:
            buckets[m.context_profile] = []
        buckets[m.context_profile].append(m)

    for bucket_label, results in buckets.items():
        max_ctx = max(m.context_tokens for m in results)
        scenario = ScenarioResult(
            num_users=num_users,
            context_profile=bucket_label,
            context_tokens=max_ctx,
            wall_time_s=_compute_bucket_wall_time(results, num_users),
            requests=results,
        )
        run.scenarios.append(scenario)

        stats = analyze_scenario(scenario)
        _print_bucket_stats(bucket_label, stats, num_users=num_users, scenario=scenario)


def _print_bucket_stats(
    label: str,
    stats,
    *,
    num_users: int = 1,
    scenario: ScenarioResult | None = None,
) -> None:
    if stats.successful == 0:
        console.print(f"\n  [red]{label}: all {stats.total_requests} requests failed[/red]")
        if scenario and any(is_context_length_error(r.error) for r in scenario.failures):
            console.print(
                "         [yellow]↳ prompt exceeded model's context window[/yellow]"
            )
        return

    user_info = f" ({num_users}u)" if num_users > 1 else ""
    console.print(
        f"\n  {label}{user_info}: {stats.successful}/{stats.total_requests} ok | "
        f"tok/s: [bold cyan]{stats.tok_per_sec.median:.1f}[/bold cyan] | "
        f"TTFT: {stats.ttft_ms.median:.0f}ms p50 | "
        f"Agg: {stats.aggregate_tok_per_sec:.0f} tok/s"
    )


def _print_replay_summary(run: BenchmarkRun, workload: Workload, num_users: int = 1) -> None:
    from rich.table import Table

    if not run.scenarios:
        console.print(
            "\n  [yellow]No requests were sent — all were skipped or filtered out.[/yellow]"
        )
        console.print(
            "  [yellow]Try a larger --model-context-length or a different workload.[/yellow]"
        )
        return

    title = f"Replay: {workload.name} -> {run.model}"
    if num_users > 1:
        title += f" ({num_users} users)"

    table = Table(title=title)
    table.add_column("Context", justify="right")
    if num_users > 1:
        table.add_column("Users", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("OK", justify="right")

    for scenario in run.scenarios:
        stats = analyze_scenario(scenario)
        if stats.successful == 0:
            row = [stats.context_profile]
            if num_users > 1:
                row.append(str(scenario.num_users))
            row += [str(stats.total_requests), "[red]FAIL[/red]", "-", f"0/{stats.total_requests}"]
            table.add_row(*row)
        else:
            row = [stats.context_profile]
            if num_users > 1:
                row.append(str(scenario.num_users))
            row += [
                str(stats.total_requests),
                f"{stats.tok_per_sec.median:.1f}",
                f"{stats.ttft_ms.median:.0f}ms",
                f"{stats.successful}/{stats.total_requests}",
            ]
            table.add_row(*row)

    console.print(table)

    ctx_len_failures = sum(
        1
        for s in run.scenarios
        for r in s.failures
        if is_context_length_error(r.error)
    )
    if ctx_len_failures:
        console.print(
            f"\n  [yellow]⚠  {ctx_len_failures} request(s) failed because the prompt"
            f" exceeded the model's context window.[/yellow]"
        )
        console.print(
            "  [yellow]   Use [bold]--model-context-length N[/bold] to skip"
            " requests that exceed N tokens, or [bold]--slice-tokens N[/bold]"
            " to cap cumulative prompt size.[/yellow]"
        )


def _print_replay_dry_run(workload: Workload, url: str, num_users: int = 1) -> None:
    console.print("\n[bold yellow]DRY RUN -- no requests will be sent[/bold yellow]\n")
    console.print(f"  Target URL: {url}")
    console.print(f"  Workload: {workload.name}")
    console.print(f"  Total requests: {workload.total_requests}")
    if num_users > 1:
        total = workload.total_requests * num_users
        console.print(f"  Users: {num_users} concurrent ({total} total)")
    console.print(f"  Experiments: {len(workload.experiment_ids)}")
    console.print(f"  Approx tokens: {workload.total_tokens_approx:,}")

    for i, entry in enumerate(workload.entries[:5]):
        tok = sum(len(m.get("content", "")) for m in entry.messages) // 4
        console.print(f"    [{entry.seq}] ~{tok:,} tokens, max_out={entry.max_tokens}")
    if workload.total_requests > 5:
        console.print(f"    ... and {workload.total_requests - 5} more")


def _save_replay_output(config: BenchmarkConfig, run: BenchmarkRun) -> None:
    output = config.output
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".json"):
        json_path = output
    elif output.endswith(".md"):
        json_path = output[:-3] + ".json"
    else:
        json_path = output + ".json"

    run.save(json_path)
    console.print(f"\n  Results saved to {json_path}")

    if output.endswith(".md"):
        from agentic_coding_bench.report.markdown import generate_report
        report = generate_report(run, json_path=json_path)
        with open(output, "w") as f:
            f.write(report)
        console.print(f"  Report saved to {output}")
