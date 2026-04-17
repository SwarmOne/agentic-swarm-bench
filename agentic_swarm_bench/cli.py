"""CLI entry point for agentic-swarm-bench."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from agentic_swarm_bench import __version__
from agentic_swarm_bench.config import (
    CONTEXT_PROFILES,
    SUITE_CONFIGS,
    build_config,
)


class DefaultGroup(click.Group):
    """Click Group that dispatches to a default subcommand when none is recognised.

    The key trick is ``ignore_unknown_options = True``: the group passes
    unrecognised tokens straight through rather than erroring, so options
    like ``-e``/``-m``/``-w`` land in the subcommand's parser unchanged.
    """

    ignore_unknown_options = True

    def __init__(
        self, *args, default: str | None = None, default_if_no_args: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._default_cmd = default
        self._default_if_no_args = default_if_no_args

    def parse_args(self, ctx, args):
        if not args and self._default_if_no_args:
            args = [self._default_cmd]
        return super().parse_args(ctx, args)

    def get_command(self, ctx, cmd_name):
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # Unknown token - treat it as the first arg of the default command.
        ctx.meta["_default_arg0"] = cmd_name
        return super().get_command(ctx, self._default_cmd)

    def resolve_command(self, ctx, args):
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        if "_default_arg0" in ctx.meta:
            remaining = [ctx.meta.pop("_default_arg0")] + remaining
            cmd_name = remaining[0] if remaining else cmd_name
        return cmd_name, cmd, remaining

    def format_commands(self, ctx, formatter):
        """Mark the default command with * in the help listing."""
        commands = []
        for name in self.list_commands(ctx):
            cmd = self.commands.get(name)
            if cmd is None or cmd.hidden:
                continue
            label = name + ("*" if name == self._default_cmd else "")
            commands.append((label, cmd.get_short_help_str(limit=formatter.width)))
        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)

console = Console()


def _reject_users_flag(value):
    """Hard-fail any `asb replay --users N` invocation with a migration hint.

    Kept as a hidden flag so the error message is precise instead of Click's
    generic "no such option". See CHANGELOG for why `--users` was removed.
    """
    if value is None:
        return value
    raise click.UsageError(
        "`--users` was removed from `asb replay`: it made every 'user' send "
        "byte-identical poisoned payloads, so only user 0 did real prefill "
        "work and users 1..N-1 rode the KV cache for free. "
        "Use `--repetitions N --max-concurrent N` for N concurrent "
        "executions, each with a distinct poison seed."
    )


def _merge_extra_body(extra_body_json: str | None, enable_thinking: bool) -> dict | None:
    """Parse --extra-body JSON and merge with --enable-thinking convenience flag.

    Returns ``None`` when neither is set so we don't put an empty dict in the
    request payload.
    """
    import json as _json

    out: dict = {}
    if extra_body_json:
        try:
            parsed = _json.loads(extra_body_json)
        except _json.JSONDecodeError as e:
            raise click.UsageError(f"--extra-body is not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise click.UsageError("--extra-body must be a JSON object")
        out.update(parsed)

    if enable_thinking:
        existing = out.get("chat_template_kwargs") or {}
        if not isinstance(existing, dict):
            raise click.UsageError(
                "--extra-body.chat_template_kwargs must be an object to "
                "combine with --enable-thinking"
            )
        existing["enable_thinking"] = True
        out["chat_template_kwargs"] = existing

    return out or None


def _require_endpoint_model(cfg_endpoint: str, cfg_model: str) -> None:
    """Raise a clear UsageError if endpoint or model are still unset after config resolution."""
    if not cfg_endpoint:
        raise click.UsageError(
            "Missing '--endpoint' / '-e'. Pass it on the CLI, set ASB_ENDPOINT, "
            "or add 'endpoint' to a --config YAML file."
        )
    if not cfg_model:
        raise click.UsageError(
            "Missing '--model' / '-m'. Pass it on the CLI, set ASB_MODEL, "
            "or add 'model' to a --config YAML file."
        )


@click.group(cls=DefaultGroup, default="replay", default_if_no_args=True)
@click.version_option(version=__version__, prog_name="agentic-swarm-bench")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="YAML config file (overridden by CLI args and env vars)",
)
@click.pass_context
def main(ctx, config):
    """AgenticSwarmBench - Benchmark LLM inference under agentic scenarios.

    \b
    Modes:
      speed          - Measure inference speed (TTFT, tok/s, ITL, prefill)
      eval           - Evaluate code correctness (syntax, execution, functional)
      agent          - Full agentic session benchmark through recording proxy
      record         - Record a real coding session as a replayable JSONL
      replay         - Replay a recorded scenario against any endpoint
      report         - Generate reports from saved results
      compare        - Compare two benchmark runs side by side
      list-tasks     - Show available tasks and tiers
      list-scenarios - Show available built-in scenarios

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
@click.option("--endpoint", "-e", default=None, help="OpenAI-compatible URL (or set ASB_ENDPOINT)")
@click.option("--model", "-m", default=None, help="Model name for requests (or set ASB_MODEL)")
@click.option("--api-key", "-k", default="", help="API key (or set ASB_API_KEY)")
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
@click.option(
    "--users",
    "-u",
    type=click.IntRange(min=1),
    default=None,
    help="Number of concurrent users (must be >= 1)",
)
@click.option(
    "--max-users",
    type=click.IntRange(min=1),
    default=None,
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
@click.option(
    "--context-tokens",
    "-c",
    type=click.IntRange(min=1),
    default=None,
    help="Exact context size in tokens",
)
@click.option(
    "--tasks",
    "-t",
    default=None,
    help="Task range: p1-p25, p51-p75, trivial, expert, etc.",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=512,
    help="Max output tokens per request (CLI value overrides per-task defaults)",
)
@click.option(
    "--cache-mode",
    type=click.Choice(["allcold", "allwarm", "realistic"]),
    default=None,
    help="Cache mode: allcold (defeat cache), allwarm (allow cache), realistic (both passes)",
)
@click.option(
    "--timeout",
    type=click.FloatRange(min=0.1),
    default=300.0,
    help="Request timeout in seconds",
)
@click.option("--output", "-o", default=None, help="Save results to file (.md or .json)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format: markdown (default, also saves .json), json (only .json)",
)
@click.option(
    "--repetitions",
    "-r",
    type=click.IntRange(min=1),
    default=1,
    help="Repetitions per scenario (total samples = users × repetitions)",
)
@click.option("--random-context", is_flag=True, help="Randomize context per request")
@click.option("--dry-run", is_flag=True, help="Show what would run without sending requests")
@click.option(
    "--model-context-length",
    type=click.IntRange(min=1),
    default=None,
    help="Model's max context window in tokens. Skips any scenarios that exceed it.",
)
@click.option(
    "--extra-body",
    default=None,
    help=(
        "JSON merged into each request body. Use to pass vendor-specific fields "
        "like chat_template_kwargs.enable_thinking=true for vLLM-served Qwen/DeepSeek."
    ),
)
@click.option(
    "--enable-thinking",
    is_flag=True,
    default=False,
    help=(
        "Shortcut for --extra-body "
        "'{\"chat_template_kwargs\":{\"enable_thinking\":true}}'. "
        "Merged with --extra-body if both are given."
    ),
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
    cache_mode,
    timeout,
    output,
    output_format,
    repetitions,
    random_context,
    dry_run,
    model_context_length,
    extra_body,
    enable_thinking,
    verbose,
):
    """Benchmark inference speed against any OpenAI-compatible endpoint.

    \b
    Examples:
      asb speed -e http://localhost:8000 -m my-model --suite quick
      asb speed -e https://api.example.com/v1/chat/completions -m my-model
      asb speed -e http://localhost:8000 -m my-model -u 32 -p long
      asb speed -e http://localhost:8000 -m my-model --dry-run
      asb speed -e http://localhost:8000 -m my-model --format json -o results.json
    """
    from agentic_swarm_bench.runner.direct import run_speed_benchmark

    merged_extra = _merge_extra_body(extra_body, enable_thinking)

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
            "cache_mode": cache_mode,
            "timeout": timeout,
            "output": output,
            "output_format": output_format,
            "repetitions": repetitions,
            "random_context": random_context or None,
            "dry_run": dry_run or None,
            "verbose": verbose or None,
            "extra_body": merged_extra,
        },
    )

    _require_endpoint_model(cfg.endpoint, cfg.model)
    asyncio.run(run_speed_benchmark(cfg))


@main.command()
@click.option("--endpoint", "-e", default=None, help="OpenAI-compatible URL (or set ASB_ENDPOINT)")
@click.option("--model", "-m", default=None, help="Model name (or set ASB_MODEL)")
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
    from agentic_swarm_bench.runner.eval_runner import run_eval

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

    _require_endpoint_model(cfg.endpoint, cfg.model)
    asyncio.run(run_eval(cfg))


@main.command()
@click.option("--endpoint", "-e", default=None, help="OpenAI-compatible URL (or set ASB_ENDPOINT)")
@click.option("--model", "-m", default=None, help="Model name to report as (or set ASB_MODEL)")
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
@click.option(
    "--upstream-api",
    type=click.Choice(["openai", "anthropic"]),
    default=None,
    help="Upstream API format. Auto-detected from URL if not set "
    "(api.anthropic.com → anthropic, everything else → openai).",
)
@click.pass_context
def agent(
    ctx, endpoint, model, api_key, api_key_header, tasks,
    agent_cmd, proxy_port, output, upstream_api,
):
    """Run full agentic benchmark through the recording proxy.

    \b
    Starts a recording proxy that sits between an agent (like Claude Code)
    and your endpoint. The proxy records per-request timing metrics.

    \b
    Supports both OpenAI-compatible and Anthropic upstream endpoints:
      --upstream-api openai      → translates Anthropic → OpenAI (default)
      --upstream-api anthropic   → forwards Anthropic requests natively
    """
    from agentic_swarm_bench.runner.claude_code import run_agent_benchmark

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
            "upstream_api": upstream_api,
        },
    )

    _require_endpoint_model(cfg.endpoint, cfg.model)
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
      asb list-tasks
      asb list-tasks -t trivial
      asb list-tasks --tags typescript,rust
      asb list-tasks --format json
    """
    import json as json_mod

    from rich.table import Table

    from agentic_swarm_bench.tasks.registry import get_tasks

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    matched = get_tasks(task_range=tasks, tags=tag_list)
    if not matched:
        if tasks or tags:
            filter_parts = []
            if tasks:
                filter_parts.append(f"--tasks {tasks}")
            if tags:
                filter_parts.append(f"--tags {tags}")
            filters = ", ".join(filter_parts)
            raise click.UsageError(
                f"No tasks matched the given filter ({filters}). "
                "Run 'asb list-tasks' without filters to see all available tasks."
            )
        matched = get_tasks()

    if fmt == "json":
        print(json_mod.dumps(matched, indent=2))
        return

    table = Table(title=f"AgenticSwarmBench Tasks ({len(matched)} total)")
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
@click.option("--endpoint", "-e", default=None, help="Upstream LLM URL (or set ASB_ENDPOINT)")
@click.option("--model", "-m", default=None, help="Model name at upstream (or set ASB_MODEL)")
@click.option("--api-key", "-k", default="", help="API key for upstream")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option("--port", "-P", type=int, default=19000, help="Local proxy port (default: 19000)")
@click.option(
    "--output", "-o", default="recording.jsonl", help="Output JSONL file (default: recording.jsonl)"
)
@click.option(
    "--upstream-api",
    type=click.Choice(["openai", "anthropic"]),
    default=None,
    help="Upstream API format. Auto-detected from URL if not set "
    "(api.anthropic.com → anthropic, everything else → openai).",
)
def record(endpoint, model, api_key, api_key_header, port, output, upstream_api):
    """Record a real coding session as a replayable scenario recording.

    \b
    Starts a recording proxy that captures every request from your session
    agent (Claude Code, Cursor, etc.) into a JSONL recording file. Stop
    with Ctrl+C when done.

    \b
    Supports both OpenAI-compatible and Anthropic upstream endpoints.
    When the upstream is Anthropic, requests are forwarded natively
    (no translation) while still saving in OpenAI format for replay.

    \b
    Examples:
      asb record -e http://localhost:8000 -m my-model
      asb record -e https://api.anthropic.com -m claude-sonnet-4-20250514 \\
        -k $ANTHROPIC_API_KEY --api-key-header x-api-key
      asb record -e http://localhost:8000 -m my-model -o session.jsonl
    """
    import os as _os
    endpoint = endpoint or _os.getenv("ASB_ENDPOINT") or ""
    model = model or _os.getenv("ASB_MODEL") or ""
    _require_endpoint_model(endpoint, model)

    from agentic_swarm_bench.scenarios.recorder import run_recorder

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
@click.option("--endpoint", "-e", default=None, help="OpenAI-compatible URL (or set ASB_ENDPOINT)")
@click.option("--model", "-m", default=None, help="Model name (or set ASB_MODEL)")
@click.option("--api-key", "-k", default="", help="API key")
@click.option(
    "--api-key-header",
    default="Authorization",
    help="Header name for the API key",
)
@click.option(
    "--scenario",
    "-w",
    required=True,
    help="Scenario path: directory with scenario.json, single .jsonl, or built-in name",
)
@click.option("--output", "-o", default=None, help="Save results to file (.md or .json)")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0.1),
    default=300.0,
    help="Request timeout in seconds",
)
@click.option(
    "--slice-tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Replay until cumulative prompt tokens exceed N per task",
)
@click.option("--dry-run", is_flag=True, help="Preview without sending requests")
@click.option(
    "--users",
    "-u",
    "users_removed",
    type=int,
    default=None,
    hidden=True,
    expose_value=True,
    callback=lambda ctx, param, value: _reject_users_flag(value),
)
@click.option(
    "--model-context-length",
    type=click.IntRange(min=1),
    default=None,
    help="Model's max context window in tokens. Skips requests whose prompt exceeds it.",
)
@click.option(
    "--repetitions",
    "-r",
    type=click.IntRange(min=1),
    default=1,
    help="How many times to run each task (default: 1)",
)
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=10,
    help="Maximum tasks executing simultaneously (default: 10)",
)
@click.option(
    "--extra-body",
    default=None,
    help=(
        "JSON merged into each request body. Use to pass vendor-specific fields "
        "like chat_template_kwargs.enable_thinking=true."
    ),
)
@click.option(
    "--enable-thinking",
    is_flag=True,
    default=False,
    help=(
        "Shortcut for --extra-body "
        "'{\"chat_template_kwargs\":{\"enable_thinking\":true}}'."
    ),
)
@click.option(
    "--policy",
    type=click.Choice(["round_robin", "sequential", "random"]),
    default="round_robin",
    help="Task execution order: round_robin, sequential, or random (default: round_robin)",
)
@click.option(
    "--cache-mode",
    type=click.Choice(["realistic", "allcold", "allwarm"]),
    default="realistic",
    show_default=True,
    help=(
        "Cache mode: realistic (poison non-shared prefix, default), "
        "allcold (poison everything), allwarm (no poisoning)"
    ),
)
@click.pass_context
def replay(
    ctx,
    endpoint,
    model,
    api_key,
    api_key_header,
    scenario,
    output,
    timeout,
    slice_tokens,
    dry_run,
    users_removed,
    model_context_length,
    repetitions,
    max_concurrent,
    extra_body,
    enable_thinking,
    policy,
    cache_mode,
):
    """Replay a recorded scenario against any endpoint.

    \b
    Takes a scenario (directory with scenario.json, single JSONL, or
    built-in name) and replays each task's requests against the target
    endpoint, measuring TTFT, tok/s, and throughput.

    \b
    Cache modes:
      realistic  Poison the per-user portion of each request (default).
                 Shared prefix is preserved so it can be KV-cached;
                 unique user context is varied to defeat caching there.
                 This is the most production-realistic measurement.
      allcold    Poison all messages including shared prefix. Every
                 request defeats the cache entirely.
      allwarm    No poisoning. Requests sent as recorded; the server
                 can serve from KV cache freely.

    \b
    Use --repetitions to run each task N times. Use --policy to control
    execution order (round_robin, sequential, or random). Use
    --max-concurrent to cap how many tasks run at once. N concurrent
    independent users = --repetitions N --max-concurrent N (each
    repetition gets a distinct poison seed).

    \b
    Use --slice-tokens to cap cumulative prompt tokens per task.

    \b
    Examples:
      asb replay -e http://localhost:8000 -m my-model -w session.jsonl
      asb replay -e http://localhost:8000 -m my-model -w ./scenarios/my-scenario/
      asb replay -e URL -m MODEL -w scenario -r 3 --max-concurrent 5 --policy sequential
      asb replay -e URL -m MODEL -w scenario --cache-mode allwarm
    """
    from agentic_swarm_bench.scenarios.player import replay_scenario as _replay_scenario
    from agentic_swarm_bench.scenarios.schedule import Schedule

    merged_extra = _merge_extra_body(extra_body, enable_thinking)

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
            "extra_body": merged_extra,
        },
    )

    sched = Schedule(
        repetitions=repetitions,
        max_concurrent=max_concurrent,
        policy=policy,
    )

    _require_endpoint_model(cfg.endpoint, cfg.model)
    asyncio.run(
        _replay_scenario(
            cfg,
            scenario,
            slice_tokens=slice_tokens,
            model_context_length=model_context_length,
            schedule=sched,
            cache_mode=cache_mode,
            extra_body=merged_extra,
        )
    )


@main.command("list-scenarios")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_scenarios(fmt):
    """Show available built-in scenarios.

    \b
    Examples:
      asb list-scenarios
      asb list-scenarios --format json
    """
    import json as json_mod

    from rich.table import Table

    from agentic_swarm_bench.scenarios.registry import list_builtin_scenarios

    scenarios = list_builtin_scenarios()

    if fmt == "json":
        print(json_mod.dumps(scenarios, indent=2))
        return

    if not scenarios:
        console.print("No built-in scenarios found. Record one with: asb record -e URL -m MODEL")
        return

    table = Table(title=f"Built-in Scenarios ({len(scenarios)})")
    table.add_column("Name")
    table.add_column("Tasks", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Approx Tokens", justify="right")

    for s in scenarios:
        table.add_row(
            s.get("name", "?"),
            str(s.get("tasks", "?")),
            str(s.get("requests", "?")),
            f"{s.get('approx_tokens', 0):,}",
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
      agentic-swarm-bench report -i results.json -o report.md
      agentic-swarm-bench report -i results.json -f json
    """
    from agentic_swarm_bench.metrics.collector import BenchmarkRun

    run = BenchmarkRun.load(input)

    if fmt == "markdown":
        from agentic_swarm_bench.report.markdown import generate_report

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
    from agentic_swarm_bench.metrics.collector import BenchmarkRun
    from agentic_swarm_bench.report.markdown import generate_comparison

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
