"""Replay recorded scenarios against any endpoint (OpenAI or Anthropic).

Takes a scenario (JSON manifest with tasks, directory, or single JSONL
recording) and replays each task's requests against a target endpoint,
collecting the same metrics as ``asb speed``.

Scheduling model (see docs/SCHEDULING.md):
  - A scenario has T tasks; each replay runs every task R times.
  - That yields T*R schedule-tasks. One schedule-task is one (task,
    execution_index) pair with its recorded r_1..r_n request entries.
  - A Schedule (R, J, policy, seed) pre-generates the ordered list L of
    all T*R schedule-tasks.
  - Dispatch is a literal pool of J long-lived workers, each owning one
    persistent httpx.AsyncClient and pulling the next head of L whenever
    it finishes its current schedule-task.

Upstream API modes:
  openai     Send requests as OpenAI /v1/chat/completions (default).
  anthropic  Translate recorded OpenAI messages to Anthropic Messages API
             format, stream via /v1/messages, and parse Anthropic SSE.

Cache modes:
  realistic  Poison only the non-shared portion of each request. The LCP
             (longest common prefix) across tasks is preserved so the
             server can cache it; everything beyond the LCP is varied per
             execution to simulate unique user context. This is the default.
  allcold    Poison all messages (lcp_len=0) so every request defeats the
             KV cache entirely.
  allwarm    No poisoning. Requests are sent exactly as recorded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from agentic_swarm_bench.config import BenchmarkConfig, resolve_endpoint
from agentic_swarm_bench.metrics.collector import (
    BenchmarkRun,
    RequestMetrics,
    ScenarioResult,
    is_context_length_error,
)
from agentic_swarm_bench.metrics.stats import analyze_scenario
from agentic_swarm_bench.scenarios.poison import compute_scenario_lcp, poison_task_execution
from agentic_swarm_bench.scenarios.registry import (
    RecordingEntry,
    Scenario,
    Task,
    get_scenario,
)
from agentic_swarm_bench.scenarios.schedule import (
    Schedule,
    build_execution_queue,
    run_work_queue,
)

console = Console()


def _build_headers(config: BenchmarkConfig, upstream_api: str = "openai") -> dict:
    headers = {"Content-Type": "application/json"}
    if upstream_api == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if config.api_key:
            if config.api_key_header.lower() == "authorization":
                headers["x-api-key"] = config.api_key
            else:
                headers[config.api_key_header] = config.api_key
        return headers

    if not config.api_key:
        return headers
    if config.api_key_header.lower() == "authorization":
        headers["Authorization"] = f"Bearer {config.api_key}"
    else:
        headers[config.api_key_header] = config.api_key
    return headers


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


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
    max_retries: int = 0,
    extra_body: dict | None = None,
    upstream_api: str = "openai",
) -> RequestMetrics:
    """Replay a single recorded request and collect timing metrics.

    When *upstream_api* is ``"anthropic"``, the recorded OpenAI messages are
    translated to Anthropic Messages API format and streamed via ``/v1/messages``.
    """
    metrics = RequestMetrics(
        request_id=seq,
        user_id=user_id,
        task_id=f"replay-{seq}",
        context_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
    )

    if upstream_api == "anthropic":
        return await _replay_one_request_anthropic(
            client=client,
            url=url,
            model=model,
            headers=headers,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            metrics=metrics,
            on_complete=on_complete,
            max_retries=max_retries,
            extra_body=extra_body,
        )

    capped_tokens = min(max_tokens, 4096)
    token_limit_key = "max_completion_tokens" if "api.openai.com" in url else "max_tokens"
    payload = {
        "model": model,
        "messages": messages,
        token_limit_key: capped_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)

    usage_completion_tokens: int | None = None
    drop_stream_options = False

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        first_token_time = None
        first_thinking_time = None
        first_visible_time = None
        last_token_time = start
        token_count = 0
        thinking_token_count = 0
        metrics.itl_ms = []
        metrics.error = None
        usage_completion_tokens = None

        if drop_stream_options:
            payload.pop("stream_options", None)

        async def _do_stream():
            nonlocal usage_completion_tokens, drop_stream_options
            async with client.stream(
                "POST", url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status_code == 429:
                    body = await resp.aread()
                    metrics.error = f"HTTP 429: {body.decode()[:500]}"
                    return
                if resp.status_code != 200:
                    body = await resp.aread()
                    body_text = body.decode()[:500]
                    if "stream_options" in body_text.lower():
                        drop_stream_options = True
                        metrics.error = "RETRY_STREAM_OPTIONS"
                        return
                    metrics.error = f"HTTP {resp.status_code}: {body_text}"
                    return

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
                        if "completion_tokens" in usage:
                            usage_completion_tokens = usage["completion_tokens"]

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if not content and not reasoning:
                            continue

                        nonlocal first_token_time, first_thinking_time, first_visible_time
                        nonlocal last_token_time, token_count, thinking_token_count

                        now = time.perf_counter()
                        chunk_tokens = _estimate_tokens(content or reasoning or "")
                        if first_token_time is None:
                            first_token_time = now
                            metrics.ttft_ms = (now - start) * 1000
                        else:
                            metrics.itl_ms.append((now - last_token_time) * 1000)
                        last_token_time = now
                        token_count += chunk_tokens

                        if reasoning:
                            thinking_token_count += chunk_tokens
                            if first_thinking_time is None:
                                first_thinking_time = now
                                metrics.ttft_thinking_ms = (now - start) * 1000
                        if content and first_visible_time is None:
                            first_visible_time = now
                            metrics.ttft_visible_ms = (now - start) * 1000

        try:
            await asyncio.wait_for(_do_stream(), timeout=timeout)
        except asyncio.TimeoutError:
            metrics.error = f"Wall-clock timeout after {timeout}s"
            if on_complete:
                on_complete()
            return metrics
        except Exception as e:
            metrics.error = f"{type(e).__name__}: {str(e)[:500]}"
            if on_complete:
                on_complete()
            return metrics

        if metrics.error and attempt < max_retries:
            if metrics.error.startswith("HTTP 429"):
                backoff = 2 + attempt * 3
                await asyncio.sleep(backoff)
                continue
            if metrics.error == "RETRY_STREAM_OPTIONS":
                continue
        break

    end = time.perf_counter()
    metrics.total_time_s = end - start

    if usage_completion_tokens is not None:
        metrics.completion_tokens = usage_completion_tokens
    else:
        metrics.completion_tokens = token_count

    metrics.thinking_tokens = thinking_token_count

    effective_tokens = metrics.completion_tokens
    if effective_tokens > 0 and first_token_time is not None:
        metrics.decode_time_s = end - first_token_time
        if metrics.decode_time_s > 0:
            metrics.tok_per_sec = effective_tokens / metrics.decode_time_s

    if metrics.ttft_ms > 0 and metrics.prompt_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.prompt_tokens / (metrics.ttft_ms / 1000)
    elif metrics.ttft_ms > 0 and metrics.context_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.context_tokens / (metrics.ttft_ms / 1000)

    if on_complete:
        on_complete()
    return metrics


async def _replay_one_request_anthropic(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    metrics: RequestMetrics,
    on_complete=None,
    max_retries: int = 0,
    extra_body: dict | None = None,
) -> RequestMetrics:
    """Replay a single request via Anthropic Messages API.

    Translates the recorded OpenAI messages into Anthropic format, streams
    the response from /v1/messages, and parses Anthropic SSE events.
    """
    system_parts: list[str] = []
    conversation: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation.append({"role": role, "content": content})

    if not conversation:
        conversation.append({"role": "user", "content": ""})

    capped_tokens = min(max_tokens, 4096)
    payload: dict = {
        "model": model,
        "messages": conversation,
        "max_tokens": capped_tokens,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n".join(system_parts)
    if extra_body:
        payload.update(extra_body)

    anthropic_url = url.rstrip("/")
    if not anthropic_url.endswith("/v1/messages"):
        anthropic_url += "/v1/messages"

    def _parse_sse_line(line: str):
        """Parse one SSE data line, updating nonlocal metrics state."""
        nonlocal first_token_time, last_token_time, token_count

        if not line.startswith("data: "):
            return
        data_str = line[6:].strip()
        try:
            data_obj = json.loads(data_str)
        except json.JSONDecodeError:
            return

        event_type = data_obj.get("type", "")

        if event_type == "message_start":
            msg = data_obj.get("message", {})
            usage = msg.get("usage", {})
            metrics.prompt_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

        elif event_type == "content_block_delta":
            delta = data_obj.get("delta", {})
            text = None
            if delta.get("type") == "text_delta":
                text = delta.get("text")
            elif delta.get("type") == "thinking_delta":
                text = delta.get("thinking")
            if text:
                now = time.perf_counter()
                chunk_tokens = _estimate_tokens(text)
                if first_token_time is None:
                    first_token_time = now
                    metrics.ttft_ms = (now - start) * 1000
                else:
                    metrics.itl_ms.append((now - last_token_time) * 1000)
                last_token_time = now
                token_count += chunk_tokens

        elif event_type == "message_delta":
            usage = data_obj.get("usage", {})
            if usage.get("output_tokens"):
                token_count = usage["output_tokens"]

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        first_token_time = None
        last_token_time = start
        token_count = 0
        metrics.itl_ms = []
        metrics.error = None

        async def _do_stream():
            async with client.stream(
                "POST", anthropic_url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status_code == 429:
                    body = await resp.aread()
                    metrics.error = f"HTTP 429: {body.decode()[:500]}"
                    return
                if resp.status_code != 200:
                    body = await resp.aread()
                    metrics.error = f"HTTP {resp.status_code}: {body.decode()[:500]}"
                    return

                buf = b""
                async for chunk_bytes in resp.aiter_bytes():
                    buf += chunk_bytes
                    while b"\n" in buf:
                        raw_line, buf = buf.split(b"\n", 1)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        _parse_sse_line(line)

                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    _parse_sse_line(line)

        try:
            await asyncio.wait_for(_do_stream(), timeout=timeout)
        except asyncio.TimeoutError:
            metrics.error = f"Wall-clock timeout after {timeout}s"
            if on_complete:
                on_complete()
            return metrics
        except Exception as e:
            metrics.error = f"{type(e).__name__}: {str(e)[:500]}"
            if on_complete:
                on_complete()
            return metrics

        if metrics.error and attempt < max_retries:
            if metrics.error.startswith("HTTP 429"):
                backoff = 2 + attempt * 3
                await asyncio.sleep(backoff)
                continue
        break

    end = time.perf_counter()
    metrics.total_time_s = end - start
    metrics.completion_tokens = token_count

    if token_count > 0 and first_token_time is not None:
        metrics.decode_time_s = end - first_token_time
        if metrics.decode_time_s > 0:
            metrics.tok_per_sec = token_count / metrics.decode_time_s

    if metrics.ttft_ms and metrics.ttft_ms > 0 and metrics.prompt_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.prompt_tokens / (metrics.ttft_ms / 1000)
    elif metrics.ttft_ms and metrics.ttft_ms > 0 and metrics.context_tokens > 0:
        metrics.prefill_tok_per_sec = metrics.context_tokens / (metrics.ttft_ms / 1000)

    if on_complete:
        on_complete()
    return metrics


def _bucket_label(tokens: int) -> str:
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


def _entry_prompt_tokens(entry: RecordingEntry) -> int:
    """Best-effort prompt-token size for an entry.

    Recordings sometimes store tiny or missing ``prompt_tokens`` values when the
    upstream didn't emit usage on the stream. Using the raw field alone
    underestimates size and defeats the slice budget. Take the max of the
    recorded value and a char-based estimate so budgets always hold.
    """
    recorded = entry.prompt_tokens or 0
    estimated = sum(len(m.get("content", "")) for m in entry.messages) // 4
    return max(recorded, estimated)


def _slice_entries(
    entries: list[RecordingEntry],
    slice_tokens: int | None,
) -> list[RecordingEntry]:
    """Take entries in order until cumulative prompt tokens exceed the budget."""
    if slice_tokens is None:
        return entries

    kept: list[RecordingEntry] = []
    cumulative = 0
    for entry in entries:
        tok = _entry_prompt_tokens(entry)
        if cumulative + tok > slice_tokens and kept:
            break
        cumulative += tok
        kept.append(entry)
    return kept


def _apply_slice_to_tasks(
    tasks: list[Task],
    slice_tokens: int | None,
) -> list[Task]:
    """Return a new task list whose entries are capped at slice_tokens each."""
    if slice_tokens is None:
        return tasks
    return [
        Task(id=t.id, name=t.name, entries=_slice_entries(t.entries, slice_tokens))
        for t in tasks
    ]


def _compute_bucket_wall_time(requests: list[RequestMetrics]) -> float:
    """Wall time = max time any single slot spent on this bucket.

    We group by slot_id (stored in the ``user_id`` field for backcompat),
    sum each slot's total_time_s, and take the max. That captures the
    slowest concurrent worker, which is what actually bounds the run.
    """
    by_slot: dict[int, float] = {}
    for r in requests:
        by_slot[r.user_id] = by_slot.get(r.user_id, 0.0) + r.total_time_s
    if not by_slot:
        return 0.0
    return max(by_slot.values())


async def _replay_task_entries(
    client: httpx.AsyncClient,
    url: str,
    model_override: str,
    headers: dict,
    entries: list[RecordingEntry],
    timeout: float,
    user_id: int,
    on_complete=None,
    model_context_length: int | None = None,
    extra_body: dict | None = None,
    upstream_api: str = "openai",
) -> list[RequestMetrics]:
    """Replay one task's entries sequentially, returning all metrics."""
    results: list[RequestMetrics] = []
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
            extra_body=extra_body,
            upstream_api=upstream_api,
        )
        m.context_profile = _bucket_label(tokens)
        m.context_tokens = tokens
        results.append(m)
    return results


async def replay_scenario(
    config: BenchmarkConfig,
    scenario_path: str,
    *,
    task_filter: str | None = None,
    slice_tokens: int | None = None,
    model_context_length: int | None = None,
    schedule: Schedule | None = None,
    cache_mode: str = "realistic",
    extra_body: dict | None = None,
    json_stdout: bool = False,
    verbose: bool = False,
    upstream_api: str | None = None,
) -> BenchmarkRun:
    """Replay a recorded scenario against the configured endpoint.

    task_filter:
      Replay only the task whose ``id`` matches (default: all tasks).

    upstream_api:
      ``"openai"`` (default) or ``"anthropic"``. When ``"anthropic"``, recorded
      OpenAI messages are translated to Anthropic Messages API format and sent
      to ``/v1/messages``. Auto-detected from endpoint URL when not set.

    json_stdout:
      When True, all human-readable output goes to stderr and the final
      BenchmarkRun JSON is written to stdout for piping / machine consumption.

    verbose:
      When True, show live-updating per-task progress lines with
      phase (prefill/decode), request counts, and decode tok/s.

    cache_mode:
      realistic  Preserve shared LCP, poison divergent per-user portion (default).
      allcold    Poison everything (lcp_len=0), every request defeats the cache.
      allwarm    No poisoning, requests sent as recorded.
    """
    con = Console(stderr=True) if json_stdout else console

    from agentic_swarm_bench.proxy.utils import _detect_upstream_api
    effective_api = _detect_upstream_api(config.endpoint, upstream_api)

    scenario = get_scenario(scenario_path, task_filter=task_filter)

    if schedule is None:
        schedule = Schedule()

    if effective_api == "anthropic":
        url = config.endpoint.rstrip("/")
    else:
        url = resolve_endpoint(config.endpoint)
    headers = _build_headers(config, upstream_api=effective_api)

    sliced_tasks = _apply_slice_to_tasks(scenario.tasks, slice_tokens)
    raw_queue = build_execution_queue(sliced_tasks, schedule)
    # poison_task_execution uses (task.id, execution_index) for its RNG seed, so
    # poison output is stable regardless of queue order. Shuffling later is
    # purely about dispatch order, not about which bytes get sent.

    if cache_mode == "realistic":
        lcp_len = compute_scenario_lcp(sliced_tasks)
        execution_queue = [
            (poison_task_execution(task, lcp_len, exec_idx), exec_idx)
            for task, exec_idx in raw_queue
        ]
    elif cache_mode == "allcold":
        execution_queue = [
            (poison_task_execution(task, lcp_len=0, execution_index=exec_idx), exec_idx)
            for task, exec_idx in raw_queue
        ]
    else:
        execution_queue = raw_queue

    total_task_executions = len(execution_queue)
    total_entries = sum(t.total_requests for t, _exec_idx in execution_queue)
    task_word = "task" if len(scenario.tasks) == 1 else "tasks"

    con.print("\n[bold]AgenticSwarmBench -- replay[/bold]")
    con.print(f"  Endpoint: {config.endpoint}")
    con.print(f"  Model: {config.model}")
    if effective_api == "anthropic":
        con.print("  Upstream API: [bold magenta]anthropic[/bold magenta] (Messages API)")
    con.print(
        f"  Scenario: {scenario.name}"
        f" ({len(scenario.tasks)} {task_word}, {scenario.total_requests} requests)"
    )
    if task_filter:
        con.print(f"  Task filter: {task_filter}")
    if total_task_executions != len(scenario.tasks):
        con.print(_schedule_line(schedule))
        con.print(f"  Total executions: {total_task_executions} tasks")
    cache_mode_labels = {
        "realistic": "[yellow]realistic[/yellow] (shared prefix cached, unique context poisoned)",
        "allcold": "[red]allcold[/red] (all requests defeat cache)",
        "allwarm": "[green]allwarm[/green] (no poisoning, cache allowed)",
    }
    con.print(f"  Cache mode: {cache_mode_labels.get(cache_mode, cache_mode)}")
    if slice_tokens is not None:
        con.print(f"  Slice: ≤{slice_tokens:,} prompt tokens per recording")
    if model_context_length is not None:
        con.print(f"  Model context length: {model_context_length:,} tokens")
    con.print(f"  Approx tokens: {scenario.total_tokens_approx:,} per scenario")
    con.rule()

    if config.dry_run:
        _print_replay_dry_run(scenario, url, schedule, con=con)
        dry_run = BenchmarkRun(model=config.model, endpoint=config.endpoint)
        if json_stdout:
            sys.stdout.write(json.dumps(dry_run.to_dict(), indent=2))
            sys.stdout.write("\n")
        return dry_run

    run = BenchmarkRun(
        model=config.model,
        endpoint=config.endpoint,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    total_progress = total_entries

    # Pool of J long-lived httpx clients, one per slot. Keepalive / connection
    # reuse within a slot simulates a real user holding an open session,
    # which matters for prefix-cache measurements.
    j = max(1, min(schedule.max_concurrent, len(execution_queue)))

    # Verbose mode: track per-recording state for live-updating display
    recording_states: dict[int, _RecordingState] = {}

    if verbose:
        all_results = await _run_verbose(
            con=con,
            execution_queue=execution_queue,
            j=j,
            url=url,
            config=config,
            headers=headers,
            model_context_length=model_context_length,
            extra_body=extra_body,
            recording_states=recording_states,
            upstream_api=effective_api,
        )
    else:
        all_results = await _run_progress_bar(
            con=con,
            execution_queue=execution_queue,
            j=j,
            url=url,
            config=config,
            headers=headers,
            total_task_executions=total_task_executions,
            total_progress=total_progress,
            model_context_length=model_context_length,
            extra_body=extra_body,
            upstream_api=effective_api,
        )

    flat_results: list[RequestMetrics] = []
    for task_results in all_results:
        if task_results:
            flat_results.extend(task_results)

    buckets: dict[str, list[RequestMetrics]] = {}
    for m in flat_results:
        label = m.context_profile or "unknown"
        if label not in buckets:
            buckets[label] = []
        buckets[label].append(m)

    for bucket_label, results in buckets.items():
        max_ctx = max((m.context_tokens for m in results), default=0)
        scenario_result = ScenarioResult(
            num_users=1,
            context_profile=bucket_label,
            context_tokens=max_ctx,
            wall_time_s=_compute_bucket_wall_time(results),
            requests=results,
            cache_mode=cache_mode,
        )
        run.scenarios.append(scenario_result)

        stats = analyze_scenario(scenario_result)
        _print_bucket_stats(
            bucket_label,
            stats,
            scenario=scenario_result,
            con=con,
        )

    con.rule()
    _print_replay_summary(run, scenario, con=con)

    if config.output:
        _save_replay_output(config, run, con=con)

    if json_stdout:
        sys.stdout.write(json.dumps(run.to_dict(), indent=2))
        sys.stdout.write("\n")

    return run


# ---------------------------------------------------------------------------
# Verbose mode: live-updating per-recording progress
# ---------------------------------------------------------------------------

@dataclass
class _RecordingState:
    """Mutable state for one recording's live display line."""
    name: str = ""
    slot_id: int = 0
    phase: str = "waiting"
    current_req: int = 0
    total_reqs: int = 0
    decode_tps: float = 0.0
    done: bool = False


def _build_verbose_display(states: dict[int, _RecordingState]) -> Table:
    """Build a Rich Table for live-updating verbose output."""
    table = Table.grid(padding=(0, 1))
    table.add_column("slot", style="dim", width=8)
    table.add_column("name", width=30)
    table.add_column("phase", width=12)
    table.add_column("progress", width=14)
    table.add_column("tps", width=20)

    for _slot, st in sorted(states.items()):
        if st.done:
            phase_str = "[green]done[/green]"
            tps_str = f"[dim]{st.decode_tps:.0f} tok/s[/dim]" if st.decode_tps > 0 else ""
        elif st.phase == "prefill":
            phase_str = "[yellow][prefill][/yellow]"
            tps_str = ""
        elif st.phase == "decode":
            phase_str = "[cyan][decode][/cyan]"
            tps_str = (
                f"decode tps: [bold cyan]{st.decode_tps:.0f}[/bold cyan]"
                if st.decode_tps > 0
                else ""
            )
        else:
            phase_str = "[dim]waiting[/dim]"
            tps_str = ""

        table.add_row(
            f"slot {st.slot_id}:",
            st.name,
            phase_str,
            f"req {st.current_req}/{st.total_reqs}",
            tps_str,
        )

    return table


async def _run_verbose(
    *,
    con: Console,
    execution_queue: list,
    j: int,
    url: str,
    config: BenchmarkConfig,
    headers: dict,
    model_context_length: int | None,
    extra_body: dict | None,
    recording_states: dict[int, _RecordingState],
    upstream_api: str = "openai",
) -> list:
    """Run the replay with live-updating per-recording verbose display."""
    live = Live(
        _build_verbose_display(recording_states),
        console=con,
        refresh_per_second=4,
        transient=True,
    )

    async with contextlib.AsyncExitStack() as stack:
        slot_clients: list[httpx.AsyncClient] = [
            await stack.enter_async_context(httpx.AsyncClient())
            for _ in range(j)
        ]

        async def _run_schedule_task(
            sched_task: tuple[Task, int],
            slot_id: int,
        ) -> list[RequestMetrics]:
            task, _exec_idx = sched_task

            state = _RecordingState(
                name=task.name or task.id,
                slot_id=slot_id,
                phase="prefill",
                current_req=0,
                total_reqs=len(task.entries),
            )
            recording_states[slot_id] = state
            live.update(_build_verbose_display(recording_states))

            results: list[RequestMetrics] = []
            for i, entry in enumerate(task.entries):
                state.current_req = i + 1
                state.phase = "prefill"
                live.update(_build_verbose_display(recording_states))

                tokens = sum(len(m.get("content", "")) for m in entry.messages) // 4
                if model_context_length is not None and tokens > model_context_length:
                    continue

                model = config.model or entry.model
                m = await _replay_one_request(
                    client=slot_clients[slot_id],
                    url=url,
                    model=model,
                    headers=headers,
                    messages=entry.messages,
                    max_tokens=entry.max_tokens,
                    seq=entry.seq,
                    timeout=config.timeout,
                    user_id=slot_id,
                    extra_body=extra_body,
                    upstream_api=upstream_api,
                )
                m.context_profile = _bucket_label(tokens)
                m.context_tokens = tokens
                results.append(m)

                state.phase = "decode"
                state.decode_tps = m.tok_per_sec or 0.0
                live.update(_build_verbose_display(recording_states))

            state.done = True
            live.update(_build_verbose_display(recording_states))
            return results

        with live:
            all_results = await run_work_queue(
                execution_queue,
                _run_schedule_task,
                max_concurrent=j,
            )

    return all_results


async def _run_progress_bar(
    *,
    con: Console,
    execution_queue: list,
    j: int,
    url: str,
    config: BenchmarkConfig,
    headers: dict,
    total_task_executions: int,
    total_progress: int,
    model_context_length: int | None,
    extra_body: dict | None,
    upstream_api: str = "openai",
) -> list:
    """Run the replay with a simple aggregate progress bar (non-verbose)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=con,
        transient=True,
    ) as progress:
        task_word = "tasks" if total_task_executions != 1 else "task"
        progress_task_id = progress.add_task(
            f"Replaying ({total_task_executions} {task_word}, {total_progress} reqs)",
            total=total_progress,
        )

        async with contextlib.AsyncExitStack() as stack:
            slot_clients: list[httpx.AsyncClient] = [
                await stack.enter_async_context(httpx.AsyncClient())
                for _ in range(j)
            ]

            async def _run_schedule_task(
                sched_task: tuple[Task, int],
                slot_id: int,
            ) -> list[RequestMetrics]:
                task, _exec_idx = sched_task
                return await _replay_task_entries(
                    client=slot_clients[slot_id],
                    url=url,
                    model_override=config.model,
                    headers=headers,
                    entries=task.entries,
                    timeout=config.timeout,
                    user_id=slot_id,
                    on_complete=lambda: progress.advance(progress_task_id),
                    model_context_length=model_context_length,
                    extra_body=extra_body,
                    upstream_api=upstream_api,
                )

            all_results = await run_work_queue(
                execution_queue,
                _run_schedule_task,
                max_concurrent=j,
            )

        progress.remove_task(progress_task_id)

    return all_results


def _print_bucket_stats(
    label: str,
    stats,
    *,
    scenario: ScenarioResult | None = None,
    con: Console | None = None,
) -> None:
    c = con or console
    if stats.successful == 0:
        c.print(f"\n  [red]{label}: all {stats.total_requests} requests failed[/red]")
        if scenario and any(is_context_length_error(r.error) for r in scenario.failures):
            c.print("         [yellow]↳ prompt exceeded model's context window[/yellow]")
        return

    prefill_str = (
        f"prefill: {stats.prefill_tok_per_sec.median:.0f} tok/s | "
        if stats.prefill_tok_per_sec.count > 0
        else ""
    )
    c.print(
        f"\n  {label}: {stats.successful}/{stats.total_requests} ok | "
        f"decode: [bold cyan]{stats.tok_per_sec.median:.1f}[/bold cyan] tok/s | "
        f"{prefill_str}"
        f"TTFT: {stats.ttft_ms.median:.0f}ms p50 | "
        f"Agg: {stats.aggregate_tok_per_sec:.0f} tok/s"
    )


def _print_replay_summary(
    run: BenchmarkRun,
    scenario: Scenario,
    con: Console | None = None,
) -> None:
    c = con or console

    if not run.scenarios:
        c.print(
            "\n  [yellow]No requests were sent - all were skipped or filtered out.[/yellow]"
        )
        c.print(
            "  [yellow]Try a larger --model-context-length or a different scenario.[/yellow]"
        )
        return

    title = f"Replay: {scenario.name} -> {run.model}"

    tbl = Table(title=title)
    tbl.add_column("Context", justify="right")
    tbl.add_column("Requests", justify="right")
    tbl.add_column("Decode tok/s", justify="right")
    tbl.add_column("Prefill tok/s", justify="right")
    tbl.add_column("TTFT p50", justify="right")
    tbl.add_column("OK", justify="right")

    for s in run.scenarios:
        stats = analyze_scenario(s)
        if stats.successful == 0:
            tbl.add_row(
                stats.context_profile,
                str(stats.total_requests),
                "[red]FAIL[/red]",
                "-",
                "-",
                f"0/{stats.total_requests}",
            )
        else:
            prefill_str = (
                f"{stats.prefill_tok_per_sec.median:.0f}"
                if stats.prefill_tok_per_sec.count > 0
                else "-"
            )
            tbl.add_row(
                stats.context_profile,
                str(stats.total_requests),
                f"{stats.tok_per_sec.median:.1f}",
                prefill_str,
                f"{stats.ttft_ms.median:.0f}ms",
                f"{stats.successful}/{stats.total_requests}",
            )

    c.print(tbl)

    ctx_len_failures = sum(
        1 for s in run.scenarios for r in s.failures if is_context_length_error(r.error)
    )
    if ctx_len_failures:
        c.print(
            f"\n  [yellow]⚠  {ctx_len_failures} request(s) failed because the prompt"
            f" exceeded the model's context window.[/yellow]"
        )
        c.print(
            "  [yellow]   Use [bold]--model-context-length N[/bold] to skip"
            " requests that exceed N tokens, or [bold]--slice-tokens N[/bold]"
            " to cap cumulative prompt size.[/yellow]"
        )


def _schedule_line(schedule: Schedule) -> str:
    """Human-readable schedule summary, always including the seed when set.

    Avoids square brackets around the seed so Rich doesn't try to parse
    it as a style tag and silently drop it.
    """
    parts = [
        f"  Schedule: {schedule.policy}",
        f"× {schedule.repetitions} reps",
        f"(max {schedule.max_concurrent} concurrent)",
    ]
    if schedule.seed is not None:
        parts.append(f"seed={schedule.seed}")
    return " ".join(parts)


def _print_replay_dry_run(
    scenario: Scenario,
    url: str,
    schedule: Schedule,
    con: Console | None = None,
) -> None:
    c = con or console
    c.print("\n[bold yellow]DRY RUN -- no requests will be sent[/bold yellow]\n")
    c.print(f"  Target URL: {url}")
    c.print(f"  Scenario: {scenario.name}")
    c.print(f"  Tasks: {len(scenario.tasks)}")
    c.print(f"  Total requests: {scenario.total_requests}")
    c.print(_schedule_line(schedule))
    c.print(f"  Approx tokens: {scenario.total_tokens_approx:,}")

    for task in scenario.tasks[:5]:
        c.print(f"\n  Task: {task.id} ({task.total_requests} requests)")
        for entry in task.entries[:3]:
            tok = sum(len(m.get("content", "")) for m in entry.messages) // 4
            c.print(f"    [{entry.seq}] ~{tok:,} tokens, max_out={entry.max_tokens}")
        if task.total_requests > 3:
            c.print(f"    ... and {task.total_requests - 3} more")
    if len(scenario.tasks) > 5:
        c.print(f"\n  ... and {len(scenario.tasks) - 5} more tasks")


def _save_replay_output(
    config: BenchmarkConfig,
    run: BenchmarkRun,
    con: Console | None = None,
) -> None:
    c = con or console
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
    c.print(f"\n  Results saved to {json_path}")

    if output.endswith(".md"):
        from agentic_swarm_bench.report.markdown import generate_report

        report = generate_report(run, json_path=json_path)
        with open(output, "w") as f:
            f.write(report)
        c.print(f"  Report saved to {output}")
