"""Per-request metric collection during benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CONTEXT_LENGTH_PATTERNS = [
    "prompt is too long",
    "maximum context length",
    "context_length_exceeded",
    "token limit",
    "too many tokens",
    "exceeds the model's maximum",
    "input is too long",
    "max_prompt_length",
    "reduce your prompt",
    "maximum allowed",
]


def _percentiles(values: list[float]) -> dict:
    """Compute p50/p95/p99 + min/max/mean; all fields are always present."""
    if not values:
        return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "mean": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return round(s[idx], 2)

    mean = round(sum(s) / n, 2)
    return {
        "count": n,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "mean": mean,
    }


def is_context_length_error(error: str | None) -> bool:
    """Return True if the error string indicates the prompt exceeded the model's context window."""
    if not error:
        return False
    lower = error.lower()
    return any(p in lower for p in _CONTEXT_LENGTH_PATTERNS)


@dataclass
class RequestMetrics:
    """Timing and throughput metrics for a single streaming request.

    Field notes:
        user_id:
            In ``speed`` mode this is the concurrent synthetic-user index.
            In ``replay`` mode this is the slot_id of the pool-of-J worker
            that dispatched the request (see scenarios/schedule.py). Same
            field name is kept for JSON backcompat; ``slot_id`` below is a
            read-only alias.
        repetition_id:
            The 0-based execution_index within a schedule-task. Combined
            with ``task_id`` it uniquely identifies one schedule-task.
    """

    request_id: int = 0
    repetition_id: int = 0
    user_id: int = 0
    task_id: str = ""
    context_profile: str = ""
    context_tokens: int = 0

    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    decode_time_s: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0

    tok_per_sec: float = 0.0
    prefill_tok_per_sec: float = 0.0

    itl_ms: list[float] = field(default_factory=list)

    # Thinking/reasoning token metrics (DeepSeek R1, o3, Claude extended thinking)
    thinking_tokens: int = 0
    ttft_thinking_ms: float = 0.0
    ttft_visible_ms: float = 0.0

    error: Optional[str] = None

    @property
    def slot_id(self) -> int:
        """Alias for user_id when reading replay-mode metrics.

        Replay dispatches via a pool of J workers; user_id carries the slot
        index. Use this alias in new code to make intent explicit.
        """
        return self.user_id

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.completion_tokens > 0

    @property
    def visible_tokens(self) -> int:
        return max(0, self.completion_tokens - self.thinking_tokens)

    @property
    def thinking_overhead_ms(self) -> float:
        if self.ttft_thinking_ms > 0 and self.ttft_visible_ms > 0:
            return self.ttft_visible_ms - self.ttft_thinking_ms
        return 0.0

    @property
    def itl_p50(self) -> float:
        if not self.itl_ms:
            return 0.0
        s = sorted(self.itl_ms)
        return s[len(s) // 2]

    @property
    def itl_p95(self) -> float:
        if not self.itl_ms:
            return 0.0
        s = sorted(self.itl_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    def to_dict(self) -> dict:
        d = {
            "request_id": self.request_id,
            "repetition_id": self.repetition_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "context_profile": self.context_profile,
            "context_tokens": self.context_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_time_s": round(self.total_time_s, 3),
            "decode_time_s": round(self.decode_time_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tok_per_sec": round(self.tok_per_sec, 2),
            "prefill_tok_per_sec": round(self.prefill_tok_per_sec, 2),
            "itl_p50_ms": round(self.itl_p50, 2),
            "itl_p95_ms": round(self.itl_p95, 2),
            "error": self.error,
        }
        if self.thinking_tokens > 0:
            d["thinking_tokens"] = self.thinking_tokens
            d["ttft_thinking_ms"] = round(self.ttft_thinking_ms, 2)
            d["ttft_visible_ms"] = round(self.ttft_visible_ms, 2)
            d["thinking_overhead_ms"] = round(self.thinking_overhead_ms, 2)
        return d


@dataclass
class ScenarioResult:
    num_users: int = 0
    context_profile: str = ""
    context_tokens: int = 0
    wall_time_s: float = 0.0
    requests: list[RequestMetrics] = field(default_factory=list)
    cache_mode: str | None = None

    @property
    def successes(self) -> list[RequestMetrics]:
        return [r for r in self.requests if r.succeeded]

    @property
    def failures(self) -> list[RequestMetrics]:
        return [r for r in self.requests if not r.succeeded]

    @property
    def has_thinking(self) -> bool:
        return any(r.thinking_tokens > 0 for r in self.successes)

    def to_dict(self) -> dict:
        d = {
            "num_users": self.num_users,
            "context_profile": self.context_profile,
            "context_tokens": self.context_tokens,
            "wall_time_s": round(self.wall_time_s, 3),
            "total_requests": len(self.requests),
            "successful": len(self.successes),
            "failed": len(self.failures),
            "requests": [r.to_dict() for r in self.requests],
        }
        if self.cache_mode is not None:
            d["cache_mode"] = self.cache_mode
        return d


@dataclass
class BenchmarkRun:
    model: str = ""
    endpoint: str = ""
    started_at: str = ""
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def has_thinking(self) -> bool:
        return any(s.has_thinking for s in self.scenarios)

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "endpoint": self.endpoint,
            "started_at": self.started_at,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }
        summary = self._summary()
        if summary is not None:
            d["summary"] = summary
            d["verdict"] = summary["verdict"]
        return d

    def _summary(self) -> dict | None:
        """Aggregate verdict + percentiles across all scenarios (CI consumable)."""
        if not self.scenarios:
            return None

        # Defer import to avoid a cycle.
        from agentic_swarm_bench.metrics.stats import analyze_scenario
        from agentic_swarm_bench.report.markdown import _verdict_for_stats

        total_reqs = 0
        total_ok = 0
        total_fail = 0
        ttft_values: list[float] = []
        tok_values: list[float] = []
        prefill_values: list[float] = []
        itl_values: list[float] = []

        verdict_stats = None
        for s in self.scenarios:
            total_reqs += len(s.requests)
            total_ok += len(s.successes)
            total_fail += len(s.failures)
            for r in s.successes:
                if r.ttft_ms > 0:
                    ttft_values.append(r.ttft_ms)
                if r.tok_per_sec > 0:
                    tok_values.append(r.tok_per_sec)
                if r.prefill_tok_per_sec > 0:
                    prefill_values.append(r.prefill_tok_per_sec)
                itl_values.extend(r.itl_ms)

            stats = analyze_scenario(s)
            base = s.context_profile.split("(")[0].strip()
            if stats.successful > 0 and base in ("medium", "long"):
                if verdict_stats is None or stats.num_users < verdict_stats.num_users:
                    verdict_stats = stats

        if verdict_stats is None:
            for s in self.scenarios:
                stats = analyze_scenario(s)
                if stats.successful > 0:
                    verdict_stats = stats
                    break

        verdict = _verdict_for_stats(verdict_stats) if verdict_stats else "unknown"

        return {
            "verdict": verdict,
            "total_requests": total_reqs,
            "successful": total_ok,
            "failed": total_fail,
            "ttft_ms": _percentiles(ttft_values),
            "tok_per_sec": _percentiles(tok_values),
            "prefill_tok_per_sec": _percentiles(prefill_values),
            "itl_ms": _percentiles(itl_values),
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> BenchmarkRun:
        with open(path) as f:
            data = json.load(f)

        run = cls(
            model=data.get("model", ""),
            endpoint=data.get("endpoint", ""),
            started_at=data.get("started_at", ""),
        )
        for s_data in data.get("scenarios", []):
            scenario = ScenarioResult(
                num_users=s_data["num_users"],
                context_profile=s_data["context_profile"],
                context_tokens=s_data["context_tokens"],
                wall_time_s=s_data["wall_time_s"],
                cache_mode=s_data.get("cache_mode"),
            )
            for r_data in s_data.get("requests", []):
                m = RequestMetrics(
                    request_id=r_data.get("request_id", 0),
                    repetition_id=r_data.get("repetition_id", 0),
                    user_id=r_data.get("user_id", 0),
                    task_id=r_data.get("task_id", ""),
                    context_profile=r_data.get("context_profile", ""),
                    context_tokens=r_data.get("context_tokens", 0),
                    ttft_ms=r_data.get("ttft_ms", 0),
                    total_time_s=r_data.get("total_time_s", 0),
                    decode_time_s=r_data.get("decode_time_s", 0),
                    prompt_tokens=r_data.get("prompt_tokens", 0),
                    completion_tokens=r_data.get("completion_tokens", 0),
                    tok_per_sec=r_data.get("tok_per_sec", 0),
                    prefill_tok_per_sec=r_data.get("prefill_tok_per_sec", 0),
                    thinking_tokens=r_data.get("thinking_tokens", 0),
                    ttft_thinking_ms=r_data.get("ttft_thinking_ms", 0),
                    ttft_visible_ms=r_data.get("ttft_visible_ms", 0),
                    error=r_data.get("error"),
                )
                scenario.requests.append(m)
            run.scenarios.append(scenario)
        return run
