"""Eval mode: send tasks, collect code outputs, validate correctness."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from agentic_swarm_bench.config import BenchmarkConfig, resolve_endpoint
from agentic_swarm_bench.runner.direct import _build_headers
from agentic_swarm_bench.tasks.context.codebase_context import build_messages
from agentic_swarm_bench.tasks.registry import get_tasks

console = Console()


async def _get_completion(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
) -> str:
    """Send a non-streaming request and return the text response."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        return f"ERROR: HTTP {resp.status_code}: {resp.text[:200]}"

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return "ERROR: No choices in response"

    return choices[0].get("message", {}).get("content", "")


def _extract_code(text: str) -> str:
    """Extract Python code from model output. Tries multiple strategies."""
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced[0].strip()

    # Try indented block (common for small models that skip fences)
    indented = []
    for line in text.split("\n"):
        if line.startswith("    ") or line.startswith("\t") or line.strip() == "":
            indented.append(line)
        elif indented and any(line_.strip() for line_ in indented):
            break
        else:
            indented = []
    if indented and any(line_.strip() for line_ in indented):
        return "\n".join(indented).strip()

    # Strip leading prose lines before code (e.g., "Here is the code:")
    lines = text.strip().split("\n")
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (
            stripped.startswith(("def ", "class ", "import ", "from ", "print(", "#"))
            or re.match(r"^[a-zA-Z_]\w*\s*[=(]", stripped)
        ):
            code_start = i
            break

    if code_start > 0:
        return "\n".join(lines[code_start:]).strip()

    return text.strip()


def validate_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code parses without syntax errors."""
    try:
        compile(code, "<eval>", "exec")
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def validate_execution(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Check if Python code runs without errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, "OK"
            return False, f"Exit {result.returncode}: {result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Timed out"
        except Exception as e:
            return False, str(e)
        finally:
            Path(f.name).unlink(missing_ok=True)


async def run_eval(config: BenchmarkConfig) -> None:
    """Run the evaluation benchmark."""
    tasks = get_tasks(task_range=config.task_range)
    if not tasks:
        tasks = get_tasks(task_range="p1-p25")

    ctx_tokens = config.context_tokens or 6000
    url = resolve_endpoint(config.endpoint)
    headers = _build_headers(config)

    console.print("\n[bold]agentic-swarm-bench eval[/bold]")
    console.print(f"  Endpoint: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Tasks: {len(tasks)}")
    console.print(f"  Validation: {config.validate}")
    console.print(f"  Context: {ctx_tokens} tokens")
    console.rule()

    results = []

    async with httpx.AsyncClient() as client:
        for i, task in enumerate(tasks):
            console.print(
                f"  [{i + 1}/{len(tasks)}] {task['id']}: {task['prompt'][:60]}...", end=" "
            )

            messages = build_messages(task["prompt"], ctx_tokens, defeat_cache=False)
            max_tok = task.get("max_output_tokens", config.max_output_tokens)

            response = await _get_completion(
                client,
                url,
                config.model,
                headers,
                messages,
                max_tok,
                config.timeout,
            )

            if response.startswith("ERROR:"):
                console.print(f"[red]{response}[/red]")
                results.append(
                    {
                        "task": task["id"],
                        "tier": task["tier"],
                        "passed": False,
                        "error": response,
                    }
                )
                continue

            code = _extract_code(response)

            passed = False
            detail = ""

            if config.validate in ("syntax", "execution", "functional"):
                passed, detail = validate_syntax(code)

            if passed and config.validate in ("execution", "functional"):
                passed, detail = validate_execution(code)

            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"{status} {detail[:60]}")

            results.append(
                {
                    "task": task["id"],
                    "tier": task["tier"],
                    "passed": passed,
                    "detail": detail,
                }
            )

    console.rule()
    _print_eval_summary(results)

    if config.output:
        with open(config.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults saved to {config.output}")


def _print_eval_summary(results: list[dict]) -> None:
    """Print a summary table of eval results."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    rate = (passed / total * 100) if total > 0 else 0
    console.print(f"\n  Overall: {passed}/{total} passed ({rate:.0f}%)\n")

    tier_stats = {}
    for r in results:
        tier = r["tier"]
        if tier not in tier_stats:
            tier_stats[tier] = {"total": 0, "passed": 0}
        tier_stats[tier]["total"] += 1
        if r["passed"]:
            tier_stats[tier]["passed"] += 1

    table = Table(title="Results by Tier")
    table.add_column("Tier")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Rate", justify="right")

    for tier in ["trivial", "easy", "medium", "hard", "expert"]:
        if tier not in tier_stats:
            continue
        s = tier_stats[tier]
        rate = s["passed"] / s["total"] * 100 if s["total"] > 0 else 0
        table.add_row(tier, str(s["passed"]), str(s["total"]), f"{rate:.0f}%")

    console.print(table)
