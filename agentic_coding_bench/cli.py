"""CLI entry point for agentic-coding-bench."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from agentic_coding_bench import __version__
from agentic_coding_bench.config import (
    CONTEXT_PROFILES,
    SUITE_CONFIGS,
    build_config,
)

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="agentic-coding-bench")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="YAML config file (overridden by CLI args and env vars)",
)
@click.pass_context
def main(ctx, config):
    """AgenticCodingBench - Benchmark LLM inference under agentic coding workloads.

    \b
    Modes:
      speed      - Measure inference speed (TTFT, tok/s, ITL, prefill)
      eval       - Evaluate code correctness (syntax, execution, functional)
      agent      - Full agentic session benchmark through recording proxy
      report     - Generate reports from saved results
      compare    - Compare two benchmark runs side by side
      list-tasks - Show available tasks and tiers

    \b
    Endpoint URL:
      Pass any URL. If it doesn't end with /v1/chat/completions,
      the path is appended automatically:
        -e http://localhost:8000           → .../v1/chat/completions
        -e https://api.example.com/v1/chat/completions  → used as-is

    \b
    Note: Some endpoints may not return detailed error messages.
    Use --dry-run to validate your config before sending requests.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config


@main.command()
@click.option("--endpoint", "-e", required=True, help="Any OpenAI-compatible URL")
@click.option("--model", "-m", required=True, help="Model name to use in requests")
@click.option("--api-key", "-k", default="", help="API key (or set ACB_API_KEY)")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key (default: Authorization, sends 'Bearer <key>'). "
    "Set to e.g. 'X-API-Key' to send raw key in that header.",
)
@click.option(
    "--suite",
    "-s",
    type=click.Choice(list(SUITE_CONFIGS.keys())),
    help="Predefined suite: quick (6K), standard (40-70K), or full (6K-100K)",
)
@click.option("--users", "-u", type=int, default=None, help="Number of concurrent users")
@click.option(
    "--max-users", type=int, default=None,
    help="Cap max concurrent users (filters suite user lists)",
)
@click.option(
    "--context-profile",
    "-p",
    type=click.Choice(list(CONTEXT_PROFILES.keys()) + ["realistic"]),
    default=None,
    help="Context size profile: fresh=6K, short=20K, medium=40K, long=70K, "
    "full=100K, xl=200K, xxl=400K (default: realistic = sweep 6K-100K)",
)
@click.option("--context-tokens", "-c", type=int, default=None, help="Exact context size in tokens")
@click.option(
    "--tasks",
    "-t",
    default=None,
    help="Task range: p1-p25, p51-p75, trivial, expert, etc.",
)
@click.option("--max-tokens", type=int, default=512, help="Max output tokens per request")
@click.option(
    "--defeat-cache/--allow-cache",
    default=True,
    help="Defeat prefix caching (default: on)",
)
@click.option(
    "--cache-mode",
    type=click.Choice(["cold", "warm", "both"]),
    default=None,
    help="Cache test mode: cold, warm, or both (measures cache speedup)",
)
@click.option("--timeout", type=float, default=300.0, help="Request timeout in seconds")
@click.option("--output", "-o", default=None, help="Save results to file (.md or .json)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format: markdown (default, also saves .json), json (only .json)",
)
@click.option("--repetitions", "-r", type=int, default=1, help="Repetitions per scenario")
@click.option("--random-context", is_flag=True, help="Randomize context per request")
@click.option("--dry-run", is_flag=True, help="Show what would run without sending requests")
@click.option(
    "--model-context-length",
    type=int,
    default=None,
    help="Model's max context window in tokens. Skips any scenarios that exceed it.",
)
@click.option("--verbose", "-V", is_flag=True, help="Show live progress and per-request details")
@click.pass_context
def speed(
    ctx,
    endpoint,
    model,
    api_key,
    api_key_header,
    suite,
    users,
    max_users,
    context_profile,
    context_tokens,
    tasks,
    max_tokens,
    defeat_cache,
    cache_mode,
    timeout,
    output,
    output_format,
    repetitions,
    random_context,
    dry_run,
    model_context_length,
    verbose,
):
    """Benchmark inference speed against any OpenAI-compatible endpoint.

    \b
    Examples:
      acb speed -e http://localhost:8000 -m my-model --suite quick
      acb speed -e https://api.example.com/v1/chat/completions -m my-model
      acb speed -e http://localhost:8000 -m my-model -u 32 -p long
      acb speed -e http://localhost:8000 -m my-model --dry-run
      acb speed -e http://localhost:8000 -m my-model --format json -o results.json
    """
    from agentic_coding_bench.runner.direct import run_speed_benchmark

    cfg = build_config(
        config_file=ctx.obj.get("config_file"),
        cli_args={
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key or None,
            "api_key_header": api_key_header,
            "suite": suite,
            "users": users,
            "max_users": max_users,
            "context_profile": context_profile,
            "context_tokens": context_tokens,
            "model_context_length": model_context_length,
            "task_range": tasks,
            "max_output_tokens": max_tokens,
            "defeat_cache": defeat_cache,
            "cache_mode": cache_mode,
            "timeout": timeout,
            "output": output,
            "output_format": output_format,
            "repetitions": repetitions,
            "random_context": random_context or None,
            "dry_run": dry_run or None,
            "verbose": verbose or None,
        },
    )

    asyncio.run(run_speed_benchmark(cfg))


@main.command()
@click.option("--endpoint", "-e", required=True, help="Any OpenAI-compatible URL")
@click.option("--model", "-m", required=True, help="Model name")
@click.option("--api-key", "-k", default="", help="API key")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option("--tasks", "-t", default="p1-p25", help="Task range to evaluate")
@click.option(
    "--validate",
    "-v",
    type=click.Choice(["syntax", "execution", "functional"]),
    default="syntax",
    help="Validation level",
)
@click.option("--context-tokens", "-c", type=int, default=6000, help="Context size in tokens")
@click.option("--output", "-o", default=None, help="Save results to file")
@click.pass_context
def eval(ctx, endpoint, model, api_key, api_key_header, tasks, validate, context_tokens, output):
    """Evaluate code correctness of model outputs.

    \b
    Sends tasks to the endpoint, collects generated code, and validates
    at the requested level (syntax, execution, or functional correctness).
    """
    from agentic_coding_bench.runner.eval_runner import run_eval

    cfg = build_config(
        config_file=ctx.obj.get("config_file"),
        cli_args={
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key or None,
            "api_key_header": api_key_header,
            "task_range": tasks,
            "validate": validate,
            "context_tokens": context_tokens,
            "output": output,
        },
    )

    asyncio.run(run_eval(cfg))


@main.command()
@click.option("--endpoint", "-e", required=True, help="Any OpenAI-compatible URL")
@click.option("--model", "-m", required=True, help="Model name to report as")
@click.option("--api-key", "-k", default="", help="API key for upstream")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option("--tasks", "-t", default="p1-p10", help="Task range")
@click.option("--agent-cmd", default="claude", help="Agent CLI command (default: claude)")
@click.option("--proxy-port", type=int, default=19000, help="Proxy listen port")
@click.option("--output", "-o", default=None, help="Save results to file")
@click.pass_context
def agent(ctx, endpoint, model, api_key, api_key_header, tasks, agent_cmd, proxy_port, output):
    """Run full agentic benchmark through the recording proxy.

    \b
    Starts a recording proxy that sits between a coding agent (like Claude Code)
    and your endpoint. The proxy translates Anthropic API to OpenAI API and
    records per-request timing metrics.
    """
    from agentic_coding_bench.runner.claude_code import run_agent_benchmark

    cfg = build_config(
        config_file=ctx.obj.get("config_file"),
        cli_args={
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key or None,
            "api_key_header": api_key_header,
            "task_range": tasks,
            "proxy_port": proxy_port,
            "output": output,
        },
    )

    asyncio.run(run_agent_benchmark(cfg, agent_cmd=agent_cmd))


@main.command("list-tasks")
@click.option(
    "--tasks",
    "-t",
    default=None,
    help="Filter by range: p1-p25, trivial, expert, etc.",
)
@click.option("--tags", default=None, help="Filter by tags: typescript,rust,go")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_tasks(tasks, tags, fmt):
    """Show available benchmark tasks.

    \b
    Examples:
      acb list-tasks
      acb list-tasks -t trivial
      acb list-tasks --tags typescript,rust
      acb list-tasks --format json
    """
    import json as json_mod

    from rich.table import Table

    from agentic_coding_bench.tasks.registry import get_tasks

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    matched = get_tasks(task_range=tasks, tags=tag_list)
    if not matched:
        matched = get_tasks()

    if fmt == "json":
        console.print(json_mod.dumps(matched, indent=2))
        return

    table = Table(title=f"AgenticCodingBench Tasks ({len(matched)} total)")
    table.add_column("ID", justify="right")
    table.add_column("Tier")
    table.add_column("Tags")
    table.add_column("Prompt", max_width=60)
    table.add_column("Max Tokens", justify="right")

    for t in matched:
        table.add_row(
            t["id"],
            t["tier"],
            ", ".join(t.get("tags", [])),
            t["prompt"][:60] + ("..." if len(t["prompt"]) > 60 else ""),
            str(t.get("max_output_tokens", 512)),
        )

    console.print(table)


@main.command()
@click.option("--endpoint", "-e", required=True, help="Upstream LLM endpoint URL")
@click.option("--model", "-m", required=True, help="Model name at the upstream endpoint")
@click.option("--api-key", "-k", default="", help="API key for upstream")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option("--port", "-P", type=int, default=19000, help="Local proxy port (default: 19000)")
@click.option(
    "--output", "-o", default="workload.jsonl", help="Output JSONL file (default: workload.jsonl)"
)
@click.option(
    "--upstream-api",
    type=click.Choice(["openai", "anthropic"]),
    default=None,
    help="Upstream API format. Auto-detected from URL if not set "
    "(api.anthropic.com → anthropic, everything else → openai).",
)
def record(endpoint, model, api_key, api_key_header, port, output, upstream_api):
    """Record a real agentic coding session as a replayable workload.

    \b
    Starts a recording proxy that captures every request from your coding
    agent (Claude Code, Cursor, etc.) into a JSONL workload file. Stop
    with Ctrl+C when done.

    \b
    Supports both OpenAI-compatible and Anthropic upstream endpoints.
    When the upstream is Anthropic, requests are forwarded natively
    (no translation) while still saving in OpenAI format for replay.

    \b
    Examples:
      acb record -e http://localhost:8000 -m my-model
      acb record -e https://api.anthropic.com -m claude-sonnet-4-20250514 \\
        -k $ANTHROPIC_API_KEY --api-key-header x-api-key
      acb record -e http://localhost:8000 -m my-model -o session.jsonl
    """
    from agentic_coding_bench.workloads.recorder import run_recorder

    run_recorder(
        upstream_url=endpoint,
        model=model,
        api_key=api_key,
        api_key_header=api_key_header,
        port=port,
        output_file=output,
        upstream_api=upstream_api,
    )


@main.command()
@click.option("--endpoint", "-e", required=True, help="Target OpenAI-compatible URL")
@click.option("--model", "-m", required=True, help="Model name")
@click.option("--api-key", "-k", default="", help="API key")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option(
    "--workload", "-w", required=True, help="Workload file (.jsonl) or built-in name"
)
@click.option("--output", "-o", default=None, help="Save results to file (.md or .json)")
@click.option("--timeout", type=float, default=300.0, help="Request timeout in seconds")
@click.option(
    "--slice-tokens",
    type=int,
    default=None,
    help="Replay until cumulative prompt tokens exceed N",
)
@click.option("--dry-run", is_flag=True, help="Preview without sending requests")
@click.option(
    "--users", "-u", type=int, default=1,
    help="Number of concurrent users replaying the workload (default: 1)",
)
@click.pass_context
def replay(
    ctx, endpoint, model, api_key, api_key_header,
    workload, output, timeout, slice_tokens, dry_run, users,
):
    """Replay a recorded workload against any endpoint.

    \b
    Takes a JSONL workload (from `acb record` or built-in) and replays
    each request against the target endpoint, measuring TTFT, tok/s,
    and throughput.

    \b
    Use --users to simulate N concurrent users each replaying the full
    session in parallel. Each user's requests stay sequential (preserving
    natural context growth), but users run concurrently.

    \b
    Use --slice-tokens to replay only part of a session, stopping
    when cumulative prompt tokens exceed the budget. Requests run
    in their original recorded order.

    \b
    Examples:
      acb replay -e http://localhost:8000 -m my-model -w session.jsonl
      acb replay -e http://localhost:8000 -m my-model -w session.jsonl -u 4
      acb replay -e http://new-server:8000 -m my-model -w session.jsonl -o report.md
      acb replay -e URL -m MODEL -w session.jsonl --slice-tokens 1000000
    """
    from agentic_coding_bench.workloads.player import replay_workload

    cfg = build_config(
        config_file=ctx.obj.get("config_file"),
        cli_args={
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key or None,
            "api_key_header": api_key_header,
            "output": output,
            "timeout": timeout,
            "dry_run": dry_run or None,
        },
    )

    asyncio.run(replay_workload(cfg, workload, slice_tokens=slice_tokens, num_users=users))


@main.command("list-workloads")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_workloads(fmt):
    """Show available built-in workloads.

    \b
    Examples:
      acb list-workloads
      acb list-workloads --format json
    """
    import json as json_mod

    from rich.table import Table

    from agentic_coding_bench.workloads.registry import list_builtin_workloads

    workloads = list_builtin_workloads()

    if fmt == "json":
        console.print(json_mod.dumps(workloads, indent=2))
        return

    if not workloads:
        console.print(
            "No built-in workloads found. Record one with: acb record -e URL -m MODEL"
        )
        return

    table = Table(title=f"Built-in Workloads ({len(workloads)})")
    table.add_column("Name")
    table.add_column("Requests", justify="right")
    table.add_column("Experiments", justify="right")
    table.add_column("Approx Tokens", justify="right")

    for w in workloads:
        table.add_row(
            w.get("name", "?"),
            str(w.get("requests", "?")),
            str(w.get("experiments", "?")),
            f"{w.get('approx_tokens', 0):,}",
        )

    console.print(table)


@main.command()
@click.option("--input", "-i", required=True, help="Results JSON file to read")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format",
)
def report(input, output, fmt):
    """Generate a report from saved benchmark results.

    \b
    Examples:
      agentic-coding-bench report -i results.json -o report.md
      agentic-coding-bench report -i results.json -f json
    """
    from agentic_coding_bench.metrics.collector import BenchmarkRun

    run = BenchmarkRun.load(input)

    if fmt == "markdown":
        from agentic_coding_bench.report.markdown import generate_report

        text = generate_report(run)
    else:
        import json

        text = json.dumps(run.to_dict(), indent=2)

    if output:
        with open(output, "w") as f:
            f.write(text)
        console.print(f"Report saved to {output}")
    else:
        console.print(text)


@main.command()
@click.option("--baseline", "-b", required=True, help="Baseline results JSON")
@click.option("--candidate", "-c", required=True, help="Candidate results JSON")
@click.option("--output", "-o", default=None, help="Output file")
def compare(baseline, candidate, output):
    """Compare two benchmark runs side by side."""
    from agentic_coding_bench.metrics.collector import BenchmarkRun
    from agentic_coding_bench.report.markdown import generate_comparison

    run_a = BenchmarkRun.load(baseline)
    run_b = BenchmarkRun.load(candidate)
    text = generate_comparison(run_a, run_b)

    if output:
        with open(output, "w") as f:
            f.write(text)
        console.print(f"Comparison saved to {output}")
    else:
        console.print(text)


if __name__ == "__main__":
    main()
