"""Markdown report generation with SwarmOne branding, auto-insights, and verdicts."""

from __future__ import annotations

from agentic_swarm_bench.metrics.collector import BenchmarkRun
from agentic_swarm_bench.metrics.stats import ScenarioStats, analyze_scenario

SWARMONE_LOGO_URL = (
    "https://raw.githubusercontent.com/swarmone/agentic-swarm-bench/main/assets/swarmone-logo.svg"
)

LOGO_HEADER = f"""<p align="center">
  <img src="{SWARMONE_LOGO_URL}" alt="SwarmOne" width="200">
</p>

<h2 align="center">AgenticSwarmBench &mdash; Inference Benchmark Report</h2>
"""


# ---------------------------------------------------------------------------
# Thresholds for agentic swarm UX
# ---------------------------------------------------------------------------

TOKS_GOOD = 30
TOKS_OK = 15
TTFT_GOOD = 3000
TTFT_OK = 10000


def _grade(value: float, good: float, ok: float, lower_is_better: bool = True) -> str:
    if lower_is_better:
        if value <= good:
            return "good"
        if value <= ok:
            return "ok"
        return "poor"
    if value >= good:
        return "good"
    if value >= ok:
        return "ok"
    return "poor"


def _grade_icon(grade: str) -> str:
    return {"good": "🟢", "ok": "🟡", "poor": "🔴"}.get(grade, "⚪")


def _verdict_for_stats(stats: ScenarioStats) -> str:
    ttft_grade = _grade(stats.ttft_ms.median, TTFT_GOOD, TTFT_OK, lower_is_better=True)
    toks_grade = _grade(stats.tok_per_sec.median, TOKS_GOOD, TOKS_OK, lower_is_better=False)

    grades = [ttft_grade, toks_grade]
    if "poor" in grades:
        return "poor"
    if "ok" in grades:
        return "ok"
    return "good"


def _verdict_label(verdict: str) -> str:
    labels = {
        "good": "GOOD for agentic swarm scenarios",
        "ok": "MARGINAL for agentic swarm scenarios",
        "poor": "POOR for agentic swarm scenarios",
    }
    return labels.get(verdict, "UNKNOWN")


def _fmt_ms(value: float) -> str:
    """Format milliseconds with thousands separator for readability."""
    return f"{value:,.0f}"


def _fmt_tokens(value: float) -> str:
    """Format token count as a compact K-suffixed string (e.g. 8.9K, 42.1K)."""
    if value <= 0:
        return "-"
    if value < 1_000:
        return f"{value:.0f}"
    return f"{value / 1_000:.1f}K"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(run: BenchmarkRun, json_path: str | None = None) -> str:
    """Generate a full Markdown benchmark report with insights and verdicts."""
    all_stats = [(s, analyze_scenario(s)) for s in run.scenarios]

    sections = [
        LOGO_HEADER,
        _metadata_section(run, json_path=json_path),
        _verdict_section(all_stats),
        _key_findings_section(all_stats),
        _summary_table(run, all_stats),
        _context_scaling_section(all_stats),
        _concurrency_scaling_section(all_stats),
        _methodology_section(run),
    ]
    return "\n".join(s for s in sections if s)


def _metadata_section(run: BenchmarkRun, json_path: str | None = None) -> str:
    timestamp = run.started_at[:19].replace("T", " ") if run.started_at else "N/A"
    lines = [
        "### Configuration\n",
        "| Setting | Value |",
        "|:--------|:------|",
        f"| **Model** | `{run.model}` |",
        f"| **Endpoint** | `{run.endpoint}` |",
        f"| **Date** | {timestamp} UTC |",
        f"| **Scenarios** | {len(run.scenarios)} |",
    ]
    if json_path:
        lines.append(f"| **Raw results** | `{json_path}` |")
    lines.append("")
    return "\n".join(lines)


def _verdict_section(all_stats: list[tuple]) -> str:
    successful = [(s, st) for s, st in all_stats if st.successful > 0]
    if not successful:
        return "---\n\n## Verdict\n\n**No successful requests.** Check endpoint configuration.\n"

    best = None
    for _, st in successful:
        p = _base_profile(st.context_profile)
        if p in ("medium", "long"):
            if best is None or st.num_users < best.num_users:
                best = st
    if best is None:
        best = successful[0][1]

    verdict = _verdict_for_stats(best)
    icon = _grade_icon(verdict)
    label = _verdict_label(verdict)

    lines = [
        "---\n",
        "## Verdict\n",
        f"> {icon} **{label}** at `{best.context_profile}` context with {best.num_users} user(s)",
        ">",
        f"> **{_fmt_ms(best.ttft_ms.median)} ms** first token · "
        f"**{best.tok_per_sec.median:.1f}** tok/s/user · "
        f"**{best.aggregate_tok_per_sec:.0f}** tok/s aggregate\n",
    ]
    return "\n".join(lines)


def _key_findings_section(all_stats: list[tuple]) -> str:
    successful = [(s, st) for s, st in all_stats if st.successful > 0]
    if not successful:
        return ""

    findings: list[str] = []

    # TTFT scaling across context sizes
    profiles_with_ttft: dict[str, float] = {}
    for _, stats in successful:
        p = _base_profile(stats.context_profile)
        if p not in profiles_with_ttft or stats.num_users == 1:
            profiles_with_ttft[p] = stats.ttft_ms.median

    profile_order = ["fresh", "short", "medium", "long", "full", "xl", "xxl"]
    ordered = [(p, profiles_with_ttft[p]) for p in profile_order if p in profiles_with_ttft]
    if len(ordered) >= 2:
        first_p, first_t = ordered[0]
        last_p, last_t = ordered[-1]
        if first_t > 0:
            ratio = last_t / first_t
            findings.append(
                f"TTFT scales **{ratio:.1f}x** from `{first_p}` to `{last_p}` context "
                f"({_fmt_ms(first_t)}ms → {_fmt_ms(last_t)}ms)"
            )

    # Throughput range
    tok_values = [st.tok_per_sec.median for _, st in successful if st.tok_per_sec.median > 0]
    if tok_values:
        findings.append(
            f"Decode throughput ranges from **{min(tok_values):.1f}** to "
            f"**{max(tok_values):.1f}** tok/s per user across all scenarios"
        )

    # Concurrency efficiency
    single_user = {
        _base_profile(st.context_profile): st.aggregate_tok_per_sec
        for _, st in successful
        if st.num_users == 1 and st.aggregate_tok_per_sec > 0
    }
    multi_user = [
        (st.num_users, _base_profile(st.context_profile), st.aggregate_tok_per_sec)
        for _, st in successful
        if st.num_users > 1 and st.aggregate_tok_per_sec > 0
    ]
    for n_users, profile, agg in multi_user:
        if profile in single_user and single_user[profile] > 0:
            ideal = single_user[profile] * n_users
            efficiency = (agg / ideal) * 100
            findings.append(
                f"At **{n_users}x** concurrency (`{profile}`): "
                f"**{efficiency:.0f}%** scaling efficiency "
                f"({agg:.0f} vs {ideal:.0f} ideal tok/s)"
            )
            break

    # Thinking overhead
    thinkers = [(s, st) for s, st in successful if st.has_thinking]
    if thinkers:
        overhead = max(st.thinking_overhead_ms.median for _, st in thinkers)
        findings.append(f"Reasoning model detected: **{overhead:.0f}ms** thinking overhead")

    if not findings:
        return ""

    lines = ["### Key Findings\n"]
    for f in findings:
        lines.append(f"- {f}")
    lines.append("\n---\n")
    return "\n".join(lines)


def _experience_label(ttft: float, toks: float) -> str:
    """Human-readable UX description for a given TTFT and tok/s."""
    if ttft < 1000:
        ttft_feel = "instant"
    elif ttft < 3000:
        ttft_feel = "responsive"
    elif ttft < 5000:
        ttft_feel = "slight pause"
    elif ttft < 10000:
        ttft_feel = "noticeable wait"
    else:
        ttft_feel = "disruptive"

    if toks > 50:
        toks_feel = "fast streaming"
    elif toks > 30:
        toks_feel = "smooth streaming"
    elif toks > 15:
        toks_feel = "slow streaming"
    else:
        toks_feel = "sluggish"

    return f"{ttft_feel.capitalize()}, {toks_feel}"


def _summary_table(run: BenchmarkRun, all_stats: list[tuple]) -> str:
    header = (
        "| | Users | Context | Avg prompt | Tok/s (med) | TTFT p50 | TTFT p99 "
        "| ITL p50 | Agg tok/s | Output tok | Completed | Experience |"
    )
    sep = (
        "|:-:|------:|--------:|-----------:|------------:|---------:|---------:"
        "|--------:|----------:|-----------:|----------:|------------|"
    )
    lines = ["## Results\n", header, sep]

    for _, stats in all_stats:
        if stats.successful == 0:
            lines.append(
                f"| {_grade_icon('poor')} | {stats.num_users} | {stats.context_profile} "
                f"| - | FAIL | - | - | - | - | - "
                f"| 0/{stats.total_requests} | - |"
            )
            continue

        verdict = _verdict_for_stats(stats)
        prompt_str = _fmt_tokens(stats.avg_prompt_tokens)
        experience = _experience_label(stats.ttft_ms.median, stats.tok_per_sec.median)
        lines.append(
            f"| {_grade_icon(verdict)} | {stats.num_users} | {stats.context_profile} | "
            f"{prompt_str} | "
            f"**{stats.tok_per_sec.median:.1f}** | "
            f"{_fmt_ms(stats.ttft_ms.median)} ms | "
            f"{_fmt_ms(stats.ttft_ms.p99)} ms | "
            f"{stats.itl_ms.median:.1f} ms | "
            f"{stats.aggregate_tok_per_sec:.0f} | "
            f"{stats.output_tokens.median:.0f} | "
            f"{stats.successful}/{stats.total_requests} | "
            f"{experience} |"
        )

    lines.append("")

    if run.has_thinking:
        lines.extend(_thinking_section(run))

    return "\n".join(lines)


def _context_scaling_section(all_stats: list[tuple]) -> str:
    """Show how performance changes as context grows with ASCII mini-chart."""
    single_user = [(s, st) for s, st in all_stats if st.num_users == 1 and st.successful > 0]
    if len(single_user) < 2:
        return ""

    lines = ["## Context Scaling (1 user)\n"]
    lines.append("How TTFT and throughput change as context grows:\n")
    lines.append("```")

    max_ttft = max(st.ttft_ms.median for _, st in single_user)
    max_toks = max(st.tok_per_sec.median for _, st in single_user)
    bar_width = 25

    lines.append(
        f"{'Profile':<10} {'TTFT':>8}  {'':─<{bar_width}}  {'Tok/s':>7}  {'':─<{bar_width}}"
    )
    lines.append(f"{'─' * 10} {'─' * 8}  {'─' * bar_width}  {'─' * 7}  {'─' * bar_width}")

    for _, stats in single_user:
        ttft = stats.ttft_ms.median
        toks = stats.tok_per_sec.median
        profile = stats.context_profile

        ttft_len = int((ttft / max_ttft) * bar_width) if max_ttft > 0 else 0
        toks_len = int((toks / max_toks) * bar_width) if max_toks > 0 else 0

        ttft_bar = "▓" * ttft_len + "░" * (bar_width - ttft_len)
        toks_bar = "█" * toks_len + "░" * (bar_width - toks_len)

        lines.append(f"{profile:<10} {ttft:>7.0f}ms  {ttft_bar}  {toks:>6.1f}  {toks_bar}")

    lines.append("```\n")
    lines.append("▓ = TTFT (lower is better) · █ = Tok/s/user (higher is better)\n")
    return "\n".join(lines)


def _concurrency_scaling_section(all_stats: list[tuple]) -> str:
    """Show efficiency at different concurrency levels."""
    profile_groups: dict[str, list[tuple]] = {}
    for s, st in all_stats:
        if st.successful == 0:
            continue
        p = _base_profile(st.context_profile)
        if p not in profile_groups:
            profile_groups[p] = []
        profile_groups[p].append((s, st))

    sections_with_data = []
    for profile, entries in profile_groups.items():
        if len(entries) < 2:
            continue

        entries.sort(key=lambda x: x[1].num_users)
        single = next((st for _, st in entries if st.num_users == 1), None)
        if single is None:
            continue

        base_toks = single.tok_per_sec.median
        if base_toks <= 0:
            continue

        rows = []
        for _, st in entries:
            ideal_agg = base_toks * st.num_users
            actual_agg = st.aggregate_tok_per_sec
            efficiency = (actual_agg / ideal_agg * 100) if ideal_agg > 0 else 0
            rows.append((st.num_users, st.tok_per_sec.median, actual_agg, efficiency))

        sections_with_data.append((profile, rows))

    if not sections_with_data:
        return ""

    lines = ["## Concurrency Scaling\n"]

    for profile, rows in sections_with_data:
        lines.append(f"### `{profile}` context\n")
        lines.append("| Users | Tok/s/user | Agg tok/s | Efficiency |")
        lines.append("|------:|-----------:|----------:|-----------:|")

        for n_users, tps, agg, eff in rows:
            eff_icon = _grade_icon(_grade(eff, 80, 50, lower_is_better=False))
            lines.append(f"| {n_users} | {tps:.1f} | {agg:.0f} | {eff_icon} {eff:.0f}% |")
        lines.append("")

    return "\n".join(lines)


def _thinking_section(run: BenchmarkRun) -> list[str]:
    """Render thinking/reasoning token analysis."""
    lines = [
        "## Reasoning Token Analysis\n",
        (
            "The model produced reasoning/thinking tokens "
            "(e.g. DeepSeek R1, o3, Claude Extended Thinking). "
            "These are generated before visible output and affect "
            "perceived latency.\n"
        ),
        ("| Users | Context | TTFT (thinking) | TTFT (visible) | Overhead | Thinking tok |"),
        ("|------:|--------:|----------------:|---------------:|---------:|-------------:|"),
    ]

    for scenario in run.scenarios:
        stats = analyze_scenario(scenario)
        if not stats.has_thinking or stats.successful == 0:
            continue

        thinkers = [r for r in scenario.successes if r.thinking_tokens > 0]
        avg_thinking = sum(r.thinking_tokens for r in thinkers) / len(thinkers) if thinkers else 0

        lines.append(
            f"| {stats.num_users} | {stats.context_profile} | "
            f"{_fmt_ms(stats.ttft_thinking_ms.median)} ms | "
            f"{_fmt_ms(stats.ttft_visible_ms.median)} ms | "
            f"{_fmt_ms(stats.thinking_overhead_ms.median)} ms | "
            f"{avg_thinking:.0f} |"
        )

    lines.append("")
    return lines


def _methodology_section(run: BenchmarkRun) -> str:
    lines = [
        "<details>",
        "<summary><strong>Methodology</strong></summary>\n",
        "### What was measured\n",
        "| Term | Meaning |",
        "|------|---------|",
        "| **TTFT** | Time To First Token - latency before first output |",
        "| **Tok/s/user** | Decode throughput per user |",
        "| **ITL** | Inter-Token Latency - time between consecutive tokens |",
        "| **Prefill tok/s** | Input token processing rate during TTFT |",
        "| **Agg tok/s** | Aggregate throughput across all concurrent users |",
        "| **TTFT (thinking)** | Time to first *reasoning* token (R1, o3) |",
        "| **TTFT (visible)** | Time to first *visible* output token |",
        "| **Thinking overhead** | Extra latency from reasoning before output |",
        "",
        "### How it works\n",
        (
            "Each request is padded with realistic agentic swarm context "
            "- system prompts with tool schemas (Read, Write, Edit, Bash, Grep), "
            "prior conversation turns with file contents and tool results, "
            "and growing context that simulates a real coding session in "
            "Claude Code, Cursor, or Copilot.\n"
        ),
        "### Performance grades\n",
        "| Grade | TTFT | Tok/s | Meaning |",
        "|:-----:|-----:|------:|---------|",
        f"| {_grade_icon('good')} Good | < {TTFT_GOOD / 1000:.0f}s | > {TOKS_GOOD} | "
        "Responsive agentic editing experience |",
        f"| {_grade_icon('ok')} Marginal | < {TTFT_OK / 1000:.0f}s | > {TOKS_OK} | "
        "Usable but noticeable delays |",
        f"| {_grade_icon('poor')} Poor | > {TTFT_OK / 1000:.0f}s | < {TOKS_OK} | "
        "Disruptive to agentic flow |",
        "",
        "### A note on ITL (Inter-Token Latency)\n",
        (
            "ITL measures the time gap between consecutive SSE `data:` lines "
            "as received by the client. Because HTTP/2 and TCP can buffer "
            "multiple tokens into a single network frame, very low ITL values "
            "(< 1ms) typically reflect network batching rather than actual "
            "token generation intervals. Tok/s is a more reliable throughput "
            "metric. ITL is most useful for comparing relative performance "
            "across scenarios, not as an absolute measure.\n"
        ),
    ]

    lines.extend(
        [
            "### Prefix cache poisoning\n",
            (
                "Cold-start measurements use space-doubling to defeat "
                "prefix caching: isolated spaces in the context are "
                "randomly doubled, shifting BPE token boundaries and "
                "invalidating the KV cache without adding artificial "
                "content. This ensures measurements reflect true "
                "inference performance, not cache hits.\n"
            ),
        ]
    )

    lines.extend(
        [
            "</details>\n",
            "---\n",
            "*Generated by "
            "[AgenticSwarmBench]"
            "(https://github.com/swarmone/agentic-swarm-bench)"
            " · Powered by [SwarmOne](https://swarmone.ai)*\n",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def generate_comparison(run_a: BenchmarkRun, run_b: BenchmarkRun) -> str:
    """Generate a comparison report between two benchmark runs."""
    lines = [
        LOGO_HEADER,
        "## Comparison Report\n",
        "| | Baseline | Candidate |",
        "|--|----------|-----------|",
        f"| **Model** | `{run_a.model}` | `{run_b.model}` |",
        f"| **Endpoint** | `{run_a.endpoint}` | `{run_b.endpoint}` |",
        "",
    ]

    stats_a = {(s.num_users, s.context_profile): analyze_scenario(s) for s in run_a.scenarios}
    stats_b = {(s.num_users, s.context_profile): analyze_scenario(s) for s in run_b.scenarios}

    all_keys = sorted(set(stats_a.keys()) | set(stats_b.keys()))
    if not all_keys:
        lines.append("No comparable scenarios found.\n")
        return "\n".join(lines)

    lines.extend(
        [
            "### Head-to-Head\n",
            "| | Users | Context | Baseline tok/s | Candidate tok/s | Delta |",
            "|:-:|------:|--------:|---------------:|----------------:|------:|",
        ]
    )

    for key in all_keys:
        users, profile = key
        a = stats_a.get(key)
        b = stats_b.get(key)

        a_tok = a.tok_per_sec.median if a and a.successful > 0 else 0
        b_tok = b.tok_per_sec.median if b and b.successful > 0 else 0

        if a_tok > 0 and b_tok > 0:
            delta_pct = ((b_tok - a_tok) / a_tok) * 100
            delta_str = f"{delta_pct:+.1f}%"
            icon = (
                _grade_icon("good")
                if delta_pct > 5
                else (_grade_icon("poor") if delta_pct < -5 else _grade_icon("ok"))
            )
        else:
            delta_str = "N/A"
            icon = "⚪"

        lines.append(f"| {icon} | {users} | {profile} | {a_tok:.1f} | {b_tok:.1f} | {delta_str} |")

    lines.append("")
    lines.extend(_ascii_chart(all_keys, stats_a, stats_b))

    a_wins = 0
    b_wins = 0
    n_valid = 0  # scenarios where both sides had successful requests
    for key in all_keys:
        a = stats_a.get(key)
        b = stats_b.get(key)
        a_ok = a is not None and a.successful > 0
        b_ok = b is not None and b.successful > 0
        if not (a_ok and b_ok):
            continue  # skip N/A scenarios from win counting
        n_valid += 1
        a_tok = a.tok_per_sec.median
        b_tok = b.tok_per_sec.median
        if a_tok > b_tok:
            a_wins += 1
        elif b_tok > a_tok:
            b_wins += 1

    n_invalid = len(all_keys) - n_valid
    failure_note = f" ({n_invalid} scenario(s) excluded: zero completions)" if n_invalid else ""
    total_label = f"{n_valid}" if not n_invalid else f"{n_valid}/{len(all_keys)} valid"

    if a_wins > b_wins:
        lines.append(
            f"\n> **Baseline** (`{run_a.model}`) wins "
            f"**{a_wins}/{n_valid}** scenarios{failure_note}\n"
        )
    elif b_wins > a_wins:
        lines.append(
            f"\n> **Candidate** (`{run_b.model}`) wins "
            f"**{b_wins}/{n_valid}** scenarios{failure_note}\n"
        )
    elif n_valid == 0:
        lines.append(
            "\n> No valid comparison scenarios"
            " (all requests failed on one or both sides)\n"
        )
    else:
        lines.append(f"\n> **Tied** across {total_label} scenarios{failure_note}\n")

    lines.extend(
        [
            "",
            "---\n",
            "*Generated by [AgenticSwarmBench](https://github.com/swarmone/agentic-swarm-bench)*\n",
        ]
    )
    return "\n".join(lines)


def _ascii_chart(
    keys: list[tuple[int, str]],
    stats_a: dict,
    stats_b: dict,
) -> list[str]:
    """Render a simple ASCII bar chart comparing tok/s across scenarios."""
    lines = ["### Visual Comparison\n", "```"]

    pairs = []
    for key in keys:
        users, profile = key
        a = stats_a.get(key)
        b = stats_b.get(key)
        a_tok = a.tok_per_sec.median if a and a.successful > 0 else 0
        b_tok = b.tok_per_sec.median if b and b.successful > 0 else 0
        label = f"{users}u/{profile}"
        pairs.append((label, a_tok, b_tok))

    if not pairs:
        return []

    max_val = max(max(a, b) for _, a, b in pairs)
    if max_val <= 0:
        return []

    bar_width = 30
    label_width = max(len(label) for label, _, _ in pairs)

    for label, a_tok, b_tok in pairs:
        a_bar = int((a_tok / max_val) * bar_width) if max_val > 0 else 0
        b_bar = int((b_tok / max_val) * bar_width) if max_val > 0 else 0

        a_str = f"{'█' * a_bar:<{bar_width}}"
        b_str = f"{'█' * b_bar:<{bar_width}}"
        lines.append(f"  {label:>{label_width}}  Baseline  |{a_str}| {a_tok:.1f}")
        lines.append(f"  {' ' * label_width}  Candidate |{b_str}| {b_tok:.1f}")
        lines.append("")

    lines.append("```")
    return lines


def _base_profile(profile: str) -> str:
    """Strip cache pass labels like 'medium (allcold)' -> 'medium'."""
    return profile.split("(")[0].strip()
