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


def is_context_length_error(error: str | None) -> bool:
    """Return True if the error string indicates the prompt exceeded the model's context window."""
    if not error:
        return False
    lower = error.lower()
    return any(p in lower for p in _CONTEXT_LENGTH_PATTERNS)


@dataclass
class RequestMetrics:
    """Timing and throughput metrics for a single streaming request."""

    request_id: int = 0
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
        return {
            "num_users": self.num_users,
            "context_profile": self.context_profile,
            "context_tokens": self.context_tokens,
            "wall_time_s": round(self.wall_time_s, 3),
            "total_requests": len(self.requests),
            "successful": len(self.successes),
            "failed": len(self.failures),
            "requests": [r.to_dict() for r in self.requests],
        }


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
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "started_at": self.started_at,
            "scenarios": [s.to_dict() for s in self.scenarios],
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
            )
            for r_data in s_data.get("requests", []):
                m = RequestMetrics(
                    request_id=r_data.get("request_id", 0),
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
