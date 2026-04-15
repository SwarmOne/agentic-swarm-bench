"""Agent runner: orchestrates Claude Code (or similar) through the recording proxy."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.tasks.registry import get_tasks

console = Console()


async def run_agent_benchmark(
    config: BenchmarkConfig,
    agent_cmd: str = "claude",
) -> None:
    """Run agentic benchmark: start proxy, feed tasks through an agent, collect metrics."""
    if not shutil.which(agent_cmd):
        console.print(
            f"[red]Error: '{agent_cmd}' not found in PATH.[/red]\n"
            f"Install Claude Code: npm install -g @anthropic-ai/claude-code"
        )
        return

    tasks = get_tasks(task_range=config.task_range)
    if not tasks:
        tasks = get_tasks(task_range="p1-p10")

    workdir = Path(tempfile.mkdtemp(prefix="agentic-swarm-bench-"))

    console.print("\n[bold]agentic-swarm-bench agent[/bold]")
    console.print(f"  Upstream: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Agent: {agent_cmd}")
    console.print(f"  Proxy port: {config.proxy_port}")
    console.print(f"  Tasks: {len(tasks)}")
    console.print(f"  Workdir: {workdir}")

    from agentic_swarm_bench.proxy.server import _detect_upstream_api

    detected_api = _detect_upstream_api(config.endpoint, config.upstream_api)
    console.print(f"  Upstream API: {detected_api}")

    proxy_proc = _start_proxy(config, log_dir=str(workdir))
    if proxy_proc is None:
        return

    try:
        await asyncio.sleep(2)

        if not await _preflight_check(config.endpoint, detected_api):
            _stop_proxy(proxy_proc)
            return

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{config.proxy_port}"
        env["ANTHROPIC_AUTH_TOKEN"] = "agentic-swarm-bench"
        env["ANTHROPIC_MODEL"] = config.model
        env["CLAUDE_MODEL"] = config.model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        empty_count = 0
        for i, task in enumerate(tasks):
            task_dir = workdir / f"task_{task['id']}"
            task_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"\n  [{i + 1}/{len(tasks)}] {task['id']}: {task['prompt'][:70]}...")

            t_start = time.perf_counter()
            try:
                result = subprocess.run(
                    [agent_cmd, "--print", task["prompt"]],
                    cwd=str(task_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
                elapsed = time.perf_counter() - t_start
                console.print(f"    Completed in {elapsed:.1f}s (exit={result.returncode})")

                log_file = workdir / f"{task['id']}.log"
                with open(log_file, "w") as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n--- STDERR ---\n")
                        f.write(result.stderr)

                if not result.stdout.strip():
                    empty_count += 1
                    if result.stderr.strip():
                        stderr_preview = result.stderr.strip()[:200]
                        console.print(
                            f"    [dim yellow]No stdout. stderr: {stderr_preview}[/dim yellow]"
                        )

                if empty_count >= 3 and i < 3:
                    console.print(
                        "\n  [bold red]Aborting: first 3 tasks "
                        "all produced empty output.[/bold red]"
                        "\n  [red]The upstream endpoint is likely "
                        "not returning valid LLM responses."
                        "\n  Check that your endpoint serves "
                        "OpenAI-compatible /v1/chat/completions."
                        "[/red]"
                    )
                    break

            except subprocess.TimeoutExpired:
                console.print(f"    [yellow]Timed out after {config.timeout}s[/yellow]")
            except Exception as e:
                console.print(f"    [red]Error: {e}[/red]")

            await asyncio.sleep(1)

        _drain_proxy_stderr(proxy_proc, workdir)
        summary = await _fetch_and_save_summary(config.proxy_port, workdir)
        _print_results(summary, workdir, empty_count=empty_count, total_tasks=len(tasks))
        _cleanup_workdir(workdir, keep_logs=(empty_count > 0))

    finally:
        _stop_proxy(proxy_proc)


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


def _drain_proxy_stderr(proxy_proc: subprocess.Popen, workdir: Path) -> None:
    """Read any buffered proxy stderr and save it for diagnostics."""
    import select

    if proxy_proc.stderr is None:
        return
    try:
        ready, _, _ = select.select([proxy_proc.stderr], [], [], 0.5)
        if ready:
            stderr_data = proxy_proc.stderr.read1(8192).decode(errors="replace")
            if stderr_data.strip():
                proxy_log = workdir / "proxy.log"
                proxy_log.write_text(stderr_data)
                for line in stderr_data.strip().splitlines()[:5]:
                    if "error" in line.lower():
                        console.print(f"  [dim red]Proxy: {line.strip()}[/dim red]")
    except Exception:
        pass


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
            "  [yellow]No metrics.jsonl found — proxy may not "
            "have received requests[/yellow]"
        )

    if empty_count and total_tasks:
        console.print(
            f"  [yellow]Empty output: {empty_count}/{total_tasks} tasks produced no stdout[/yellow]"
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
    for task_dir in workdir.glob("task_*"):
        if task_dir.is_dir() and not any(task_dir.iterdir()):
            task_dir.rmdir()


def _stop_proxy(proxy_proc: subprocess.Popen) -> None:
    """Gracefully stop the proxy subprocess."""
    proxy_proc.terminate()
    try:
        proxy_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proxy_proc.kill()


def _start_proxy(config: BenchmarkConfig, log_dir: str = "./traces") -> subprocess.Popen | None:
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

    proc = subprocess.Popen(
        [sys.executable, "-c", script, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    console.print(f"  Proxy started (PID {proc.pid})")
    return proc
