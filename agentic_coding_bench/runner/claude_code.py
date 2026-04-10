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

from rich.console import Console

from agentic_coding_bench.config import BenchmarkConfig
from agentic_coding_bench.tasks.registry import get_tasks

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

    console.print("\n[bold]agentic-coding-bench agent[/bold]")
    console.print(f"  Upstream: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Agent: {agent_cmd}")
    console.print(f"  Proxy port: {config.proxy_port}")
    console.print(f"  Tasks: {len(tasks)}")

    proxy_proc = _start_proxy(config)
    if proxy_proc is None:
        return

    try:
        await asyncio.sleep(2)

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{config.proxy_port}"
        env["ANTHROPIC_AUTH_TOKEN"] = "agentic-coding-bench"
        env["ANTHROPIC_MODEL"] = config.model
        env["CLAUDE_MODEL"] = config.model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        workdir = Path(tempfile.mkdtemp(prefix="agentic-coding-bench-"))

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

            except subprocess.TimeoutExpired:
                console.print(f"    [yellow]Timed out after {config.timeout}s[/yellow]")
            except Exception as e:
                console.print(f"    [red]Error: {e}[/red]")

            await asyncio.sleep(1)

        console.print(f"\n  Workdir: {workdir}")
        console.print(f"  Proxy metrics: http://localhost:{config.proxy_port}/benchmark/summary")

    finally:
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()


def _start_proxy(config: BenchmarkConfig) -> subprocess.Popen | None:
    """Start the recording proxy as a subprocess."""
    try:
        import agentic_coding_bench.proxy.server  # noqa: F401
    except ImportError:
        console.print("[red]Proxy deps missing. Run: pip install agentic-coding-bench[proxy][/red]")
        return None

    import sys

    script = (
        "import json, sys; "
        "from agentic_coding_bench.proxy.server import run_proxy; "
        "args = json.loads(sys.argv[1]); "
        "run_proxy(**args)"
    )
    args_json = json.dumps(
        {
            "upstream_url": config.endpoint,
            "port": config.proxy_port,
            "model": config.model,
            "api_key": config.api_key,
            "defeat_cache": config.defeat_cache,
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", script, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    console.print(f"  Proxy started (PID {proc.pid})")
    return proc
