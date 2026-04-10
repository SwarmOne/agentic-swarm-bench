"""Statistical analysis for benchmark results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from agentic_coding_bench.metrics.collector import ScenarioResult


@dataclass
class DistributionStats:
    count: int = 0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stdev: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def compute_distribution(values: list[float]) -> DistributionStats:
    if not values:
        return DistributionStats()

    s = sorted(values)
    n = len(s)

    return DistributionStats(
        count=n,
        min=s[0],
        max=s[-1],
        mean=statistics.mean(s),
        median=statistics.median(s),
        p5=s[max(0, int(n * 0.05))],
        p95=s[min(n - 1, int(n * 0.95))],
        p99=s[min(n - 1, int(n * 0.99))],
        stdev=statistics.stdev(s) if n > 1 else 0.0,
    )


@dataclass
class ScenarioStats:
    num_users: int = 0
    context_profile: str = ""
    context_tokens: int = 0
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    wall_time_s: float = 0.0

    tok_per_sec: DistributionStats = None
    ttft_ms: DistributionStats = None
    itl_ms: DistributionStats = None
    prefill_tok_per_sec: DistributionStats = None
    output_tokens: DistributionStats = None
    total_time_s: DistributionStats = None

    # Thinking token stats (populated only when thinking tokens detected)
    ttft_thinking_ms: DistributionStats = None
    ttft_visible_ms: DistributionStats = None
    thinking_overhead_ms: DistributionStats = None
    has_thinking: bool = False

    aggregate_tok_per_sec: float = 0.0
    actual_prompt_tokens: int = 0
    avg_prompt_tokens: float = 0.0

    def __post_init__(self):
        empty = DistributionStats()
        for f in (
            "tok_per_sec",
            "ttft_ms",
            "itl_ms",
            "prefill_tok_per_sec",
            "output_tokens",
            "total_time_s",
            "ttft_thinking_ms",
            "ttft_visible_ms",
            "thinking_overhead_ms",
        ):
            if getattr(self, f) is None:
                setattr(self, f, empty)


def analyze_scenario(scenario: ScenarioResult) -> ScenarioStats:
    successes = scenario.successes

    all_itl = []
    for r in successes:
        all_itl.extend(r.itl_ms)

    total_tokens = sum(r.completion_tokens for r in successes)
    actual_prompt = max((r.prompt_tokens for r in successes), default=0)

    prompt_values = [r.prompt_tokens for r in successes if r.prompt_tokens > 0]
    avg_prompt = statistics.mean(prompt_values) if prompt_values else 0.0

    thinkers = [r for r in successes if r.thinking_tokens > 0]
    has_thinking = len(thinkers) > 0

    stats = ScenarioStats(
        num_users=scenario.num_users,
        context_profile=scenario.context_profile,
        context_tokens=scenario.context_tokens,
        total_requests=len(scenario.requests),
        successful=len(successes),
        failed=len(scenario.failures),
        wall_time_s=scenario.wall_time_s,
        tok_per_sec=compute_distribution([r.tok_per_sec for r in successes if r.tok_per_sec > 0]),
        ttft_ms=compute_distribution([r.ttft_ms for r in successes if r.ttft_ms > 0]),
        itl_ms=(compute_distribution(all_itl) if all_itl else DistributionStats()),
        prefill_tok_per_sec=compute_distribution(
            [r.prefill_tok_per_sec for r in successes if r.prefill_tok_per_sec > 0]
        ),
        output_tokens=compute_distribution([float(r.completion_tokens) for r in successes]),
        total_time_s=compute_distribution([r.total_time_s for r in successes]),
        has_thinking=has_thinking,
        aggregate_tok_per_sec=(
            total_tokens / scenario.wall_time_s if scenario.wall_time_s > 0 else 0
        ),
        actual_prompt_tokens=actual_prompt,
        avg_prompt_tokens=avg_prompt,
    )

    if has_thinking:
        stats.ttft_thinking_ms = compute_distribution(
            [r.ttft_thinking_ms for r in thinkers if r.ttft_thinking_ms > 0]
        )
        stats.ttft_visible_ms = compute_distribution(
            [r.ttft_visible_ms for r in thinkers if r.ttft_visible_ms > 0]
        )
        stats.thinking_overhead_ms = compute_distribution(
            [r.thinking_overhead_ms for r in thinkers if r.thinking_overhead_ms > 0]
        )

    return stats
