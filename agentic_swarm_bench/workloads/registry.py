"""Load, list, and filter recorded workloads.

A workload is a JSONL file where each line is a recorded request from a
real agentic coding session. Workloads can be:
  - Built-in (shipped with the package in workloads/data/)
  - User-recorded (via `asb record`)
  - Downloaded from the community
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "data"


@dataclass
class WorkloadEntry:
    """One recorded request in a workload."""

    seq: int = 0
    experiment_id: str = ""
    timestamp: str = ""
    messages: list[dict] = field(default_factory=list)
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 1.0
    stream: bool = True

    prompt_tokens: int = 0
    ttft_ms: float | None = None
    total_time_s: float | None = None
    completion_tokens: int = 0
    tok_per_sec: float | None = None


@dataclass
class Workload:
    """A collection of recorded requests from one or more experiments."""

    name: str = ""
    path: str = ""
    entries: list[WorkloadEntry] = field(default_factory=list)

    @property
    def experiment_ids(self) -> list[str]:
        return list({e.experiment_id for e in self.entries if e.experiment_id})

    @property
    def total_requests(self) -> int:
        return len(self.entries)

    @property
    def total_tokens_approx(self) -> int:
        return sum(
            sum(len(m.get("content", "")) for m in e.messages) // 4
            for e in self.entries
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "experiments": len(self.experiment_ids),
            "requests": self.total_requests,
            "approx_tokens": self.total_tokens_approx,
        }


def load_workload(path: str | Path) -> Workload:
    """Load a workload from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workload not found: {p}")

    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                WorkloadEntry(
                    seq=data.get("seq", 0),
                    experiment_id=data.get("experiment_id", ""),
                    timestamp=data.get("timestamp", ""),
                    messages=data.get("messages", []),
                    model=data.get("model", ""),
                    max_tokens=data.get("max_tokens", 4096),
                    temperature=data.get("temperature", 1.0),
                    stream=data.get("stream", True),
                    prompt_tokens=data.get("prompt_tokens", 0),
                    ttft_ms=data.get("ttft_ms"),
                    total_time_s=data.get("total_time_s"),
                    completion_tokens=data.get("completion_tokens", 0),
                    tok_per_sec=data.get("tok_per_sec"),
                )
            )

    return Workload(
        name=p.stem,
        path=str(p),
        entries=entries,
    )


def list_builtin_workloads() -> list[dict]:
    """List all built-in workloads shipped with the package."""
    if not BUILTIN_DIR.exists():
        return []

    workloads = []
    for f in sorted(BUILTIN_DIR.glob("*.jsonl")):
        try:
            wl = load_workload(f)
            workloads.append(wl.summary())
        except Exception:
            workloads.append({"name": f.stem, "path": str(f), "error": "Failed to load"})

    return workloads


def get_workload(name_or_path: str) -> Workload:
    """Load a workload by name (built-in) or file path."""
    p = Path(name_or_path)
    if p.exists():
        return load_workload(p)

    builtin = BUILTIN_DIR / f"{name_or_path}.jsonl"
    if builtin.exists():
        return load_workload(builtin)

    raise FileNotFoundError(
        f"Workload '{name_or_path}' not found. "
        f"Provide a path to a .jsonl file or a built-in workload name."
    )
