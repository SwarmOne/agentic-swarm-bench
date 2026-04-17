"""Agent runner: orchestrates Claude Code (or similar) through the recording proxy.

The runner models the workload exactly like ``asb replay``:

    schedule-task = (task, execution_index)
        One CLI invocation of the agent on one task.

    With T tasks and R repetitions, there are T*R schedule-tasks. A
    Schedule (R, J, policy, seed) orders them into a single pending list
    L; then a pool of J parallel subprocess workers pulls items off L
    until drained. Nothing ever waits for a batch peer.

This matches Mike's work-queue model from docs/SCHEDULING.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.scenarios.schedule import (
    Schedule,
    build_execution_queue,
    run_work_queue,
)
from agentic_swarm_bench.tasks.registry import get_tasks

console = Console()


async def run_agent_benchmark(
    config: BenchmarkConfig,
    agent_cmd: str = "claude",
    schedule: Schedule | None = None,
) -> None:
    """Run agentic benchmark: start proxy, feed a scheduled workload through agents.

    ``schedule`` controls (R, J, policy, seed). When omitted, defaults to
    one repetition per task, one worker at a time, sequential order --
    matching pre-3.2.0 behavior.
    """
    if not shutil.which(agent_cmd):
        console.print(
            f"[red]Error: '{agent_cmd}' not found in PATH.[/red]\n"
            f"Install Claude Code: npm install -g @anthropic-ai/claude-code"
        )
        return

    if schedule is None:
        schedule = Schedule(repetitions=1, max_concurrent=1, policy="sequential")

    tasks = get_tasks(task_range=config.task_range)
    if not tasks:
        tasks = get_tasks(task_range="p1-p10")

    workdir = Path(tempfile.mkdtemp(prefix="agentic-swarm-bench-"))

    execution_queue = build_execution_queue(tasks, schedule)
    total_schedule_tasks = len(execution_queue)

    console.print("\n[bold]agentic-swarm-bench agent[/bold]")
    console.print(f"  Upstream: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Agent: {agent_cmd}")
    console.print(f"  Proxy port: {config.proxy_port}")
    console.print(f"  Tasks: {len(tasks)}")
    console.print(f"  Schedule: {schedule.policy}"
                  f" × {schedule.repetitions} reps"
                  f" (max {schedule.max_concurrent} parallel agents)")
    if schedule.seed is not None:
        console.print(f"  Seed: {schedule.seed}")
    console.print(f"  Total schedule-tasks: {total_schedule_tasks}")
    console.print(f"  Workdir: {workdir}")

    from agentic_swarm_bench.proxy.server import _detect_upstream_api

    detected_api = _detect_upstream_api(config.endpoint, config.upstream_api)
    console.print(f"  Upstream API: {detected_api}")

    proxy_proc = await _start_proxy(config, log_dir=str(workdir))
    if proxy_proc is None:
        return

    try:
        await asyncio.sleep(2)

        if not await _preflight_check(config.endpoint, detected_api):
            await _stop_proxy(proxy_proc)
            return

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{config.proxy_port}"
        env["ANTHROPIC_AUTH_TOKEN"] = "agentic-swarm-bench"
        env["ANTHROPIC_MODEL"] = config.model
        env["CLAUDE_MODEL"] = config.model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        state = _AgentRunState()

        async def _run_schedule_task(
            sched_task: tuple[dict, int],
            slot_id: int,
        ) -> None:
            task, exec_idx = sched_task
            await _run_one_agent_task(
                task=task,
                exec_idx=exec_idx,
                slot_id=slot_id,
                total=total_schedule_tasks,
                agent_cmd=agent_cmd,
                env=env,
                workdir=workdir,
                timeout=config.timeout,
                state=state,
            )

        await run_work_queue(
            execution_queue,
            _run_schedule_task,
            max_concurrent=schedule.max_concurrent,
        )

        _drain_proxy_stderr(proxy_proc, workdir)
        summary = await _fetch_and_save_summary(config.proxy_port, workdir)
        _print_results(
            summary,
            workdir,
            empty_count=state.empty_count,
            total_tasks=total_schedule_tasks,
        )
        _cleanup_workdir(workdir, keep_logs=(state.empty_count > 0))

    finally:
        await _stop_proxy(proxy_proc)


class _AgentRunState:
    """Mutable counters shared across concurrent agent workers."""

    def __init__(self) -> None:
        self.empty_count = 0
        self.completed = 0
        self._lock = asyncio.Lock()

    async def record_completion(self, empty: bool) -> int:
        async with self._lock:
            self.completed += 1
            if empty:
                self.empty_count += 1
            return self.completed


async def _run_one_agent_task(
    *,
    task: dict,
    exec_idx: int,
    slot_id: int,
    total: int,
    agent_cmd: str,
    env: dict,
    workdir: Path,
    timeout: float,
    state: _AgentRunState,
) -> None:
    """Launch one agent subprocess for one schedule-task and capture its output."""
    task_dir = workdir / f"slot{slot_id}_{task['id']}_r{exec_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)

    label = f"{task['id']}#r{exec_idx}"
    preview = task["prompt"][:70]
    console.print(f"\n  [slot {slot_id}] start {label}: {preview}...")

    t_start = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            agent_cmd,
            "--print",
            task["prompt"],
            cwd=str(task_dir),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        console.print(f"    [red][slot {slot_id}] spawn failed: {e}[/red]")
        await state.record_completion(empty=True)
        return

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        returncode = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        elapsed = time.perf_counter() - t_start
        console.print(
            f"    [slot {slot_id}] [yellow]{label} timed out after {elapsed:.0f}s[/yellow]"
        )
        await state.record_completion(empty=True)
        return

    elapsed = time.perf_counter() - t_start

    log_file = workdir / f"slot{slot_id}_{task['id']}_r{exec_idx}.log"
    log_file.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

    empty = not stdout.strip()
    completed = await state.record_completion(empty=empty)

    status = f"exit={returncode}"
    if empty and stderr.strip():
        preview_err = stderr.strip()[:150]
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} "
            f"{elapsed:.1f}s [dim yellow](empty stdout; stderr: {preview_err})[/dim yellow]"
        )
    elif empty:
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} "
            f"{elapsed:.1f}s [yellow](empty stdout)[/yellow]"
        )
    else:
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} {elapsed:.1f}s"
        )


async def _preflight_check(endpoint: str, upstream_api: str) -> bool:
    """Verify the upstream endpoint is reachable and serves the expected API."""
    if upstream_api == "anthropic":
        url = f"{endpoint.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "preflight-check",
        }
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
    else:
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }

    console.print(f"\n  Pre-flight check: {url}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 404:
                alt = "openai" if upstream_api == "anthropic" else "anthropic"
                console.print(
                    f"  [bold red]FAILED: {url} returned 404[/bold red]\n"
                    f"  [red]The endpoint does not serve this path.\n"
                    f"  Try --upstream-api={alt} if the endpoint "
                    f"uses a different API format.[/red]"
                )
                return False
            console.print(f"  [green]OK[/green] (HTTP {resp.status_code})")
            return True
    except httpx.ConnectError:
        console.print(
            f"  [bold red]FAILED: Cannot connect to {endpoint}[/bold red]\n"
            f"  [red]Is the server running?[/red]"
        )
        return False
    except Exception as e:
        console.print(
            f"  [yellow]Warning: pre-flight check failed ({e}), "
            f"continuing anyway[/yellow]"
        )
        return True


def _drain_proxy_stderr(proxy_proc: asyncio.subprocess.Process, workdir: Path) -> None:
    """Read any buffered proxy stderr and save it for diagnostics."""
    if proxy_proc.stderr is None:
        return
    try:
        stderr_data = b""
        while True:
            try:
                chunk = proxy_proc.stderr._buffer[:8192]  # type: ignore[attr-defined]
            except AttributeError:
                break
            if not chunk:
                break
            stderr_data += chunk
            break
    except Exception:
        stderr_data = b""

    text = stderr_data.decode(errors="replace")
    if not text.strip():
        return
    proxy_log = workdir / "proxy.log"
    proxy_log.write_text(text)
    for line in text.strip().splitlines()[:5]:
        if "error" in line.lower():
            console.print(f"  [dim red]Proxy: {line.strip()}[/dim red]")


async def _fetch_and_save_summary(proxy_port: int, workdir: Path) -> dict | None:
    """Fetch the proxy summary while the proxy is still alive, and save to workdir."""
    url = f"http://localhost:{proxy_port}/benchmark/summary"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                summary = resp.json()
                summary_path = workdir / "summary.json"
                summary_path.write_text(json.dumps(summary, indent=2))
                return summary
    except Exception:
        pass

    metrics_path = workdir / "metrics.jsonl"
    if metrics_path.exists():
        return {"note": "Proxy unreachable, but metrics.jsonl saved", "path": str(metrics_path)}
    return None


def _print_results(
    summary: dict | None, workdir: Path, empty_count: int = 0, total_tasks: int = 0
) -> None:
    """Print a clear results section after the benchmark run."""
    console.print("\n[bold]─── Results ───[/bold]")
    console.print(f"  Workdir: {workdir}")

    metrics_path = workdir / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        line_count = len(lines)
        error_count = sum(1 for line in lines if '"error"' in line)
        console.print(f"  Metrics: {metrics_path} ({line_count} requests)")
        if error_count:
            console.print(
                f"  [red]Upstream errors: {error_count}/{line_count}"
                f" requests failed[/red]"
            )
    else:
        console.print(
            "  [yellow]No metrics.jsonl found - proxy may not "
            "have received requests[/yellow]"
        )

    if empty_count and total_tasks:
        console.print(
            f"  [yellow]Empty output: {empty_count}/{total_tasks} "
            f"schedule-tasks produced no stdout[/yellow]"
        )

    if not summary or "error" in summary:
        console.print("  [yellow]No proxy summary available[/yellow]")
        return

    total = summary.get("total_requests", 0)
    streaming = summary.get("streaming_requests", 0)
    console.print(f"  Total requests: {total}  (streaming: {streaming})")

    ttft = summary.get("ttft_ms")
    tps = summary.get("tok_per_sec")
    prefill = summary.get("prefill_tok_per_sec")

    if not any([ttft, tps, prefill]):
        console.print("  [dim]No streaming stats recorded[/dim]")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    for col in ["Count", "Min", "Mean", "Median", "P95", "Max"]:
        table.add_column(col, justify="right")

    for label, data in [("TTFT (ms)", ttft), ("Tok/s", tps), ("Prefill tok/s", prefill)]:
        if not data:
            continue
        table.add_row(
            label,
            str(data.get("count", "")),
            str(data.get("min", "")),
            str(data.get("mean", "")),
            str(data.get("median", "")),
            str(data.get("p95", "")),
            str(data.get("max", "")),
        )

    console.print(table)
    console.print(f"\n  Summary saved: {workdir / 'summary.json'}")


def _cleanup_workdir(workdir: Path, keep_logs: bool) -> None:
    """Remove debug artifacts from the workdir. Keep metrics and summary."""
    if keep_logs:
        return
    for log_file in workdir.glob("*.log"):
        log_file.unlink()
    for task_dir in workdir.glob("slot*"):
        if task_dir.is_dir() and not any(task_dir.iterdir()):
            task_dir.rmdir()


async def _stop_proxy(proxy_proc: asyncio.subprocess.Process) -> None:
    """Gracefully stop the proxy subprocess."""
    if proxy_proc.returncode is not None:
        return
    proxy_proc.terminate()
    try:
        await asyncio.wait_for(proxy_proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proxy_proc.kill()
        await proxy_proc.wait()


async def _start_proxy(
    config: BenchmarkConfig,
    log_dir: str = "./traces",
) -> asyncio.subprocess.Process | None:
    """Start the recording proxy as a subprocess."""
    try:
        import agentic_swarm_bench.proxy.server  # noqa: F401
    except ImportError:
        console.print("[red]Proxy deps missing. Run: pip install agentic-swarm-bench[proxy][/red]")
        return None

    import sys

    script = (
        "import json, sys; "
        "from agentic_swarm_bench.proxy.server import run_proxy; "
        "args = json.loads(sys.argv[1]); "
        "run_proxy(**args)"
    )
    args_json = json.dumps(
        {
            "upstream_url": config.endpoint,
            "port": config.proxy_port,
            "model": config.model,
            "api_key": config.api_key,
            "api_key_header": config.api_key_header,
            "log_dir": log_dir,
            "upstream_api": config.upstream_api,
        }
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        args_json,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    console.print(f"  Proxy started (PID {proc.pid})")
    return proc
