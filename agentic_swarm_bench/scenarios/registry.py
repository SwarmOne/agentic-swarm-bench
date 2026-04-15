"""Load, list, and filter recorded scenarios.

A scenario is a benchmark unit containing one or more tasks.
Each task has a recording -- a JSONL file where each line is a recorded
request from a real agentic coding session. Scenarios can be:
  - Built-in (shipped with the package in scenarios/data/)
  - User-recorded (via `asb record`)
  - Downloaded from the community

A scenario can be stored as:
  - A directory with a scenario.json manifest + per-task JSONL files
  - A single JSONL file (treated as a one-task scenario for backward compat)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "data"


@dataclass
class RecordingEntry:
    """One recorded request in a task's recording."""

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
class Task:
    """One logical unit of work within a scenario.

    A task groups a sequence of recorded LLM requests that form one
    coherent agentic operation (e.g. "analyze deps", "fix build").
    """

    id: str = ""
    name: str = ""
    entries: list[RecordingEntry] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        return len(self.entries)

    @property
    def total_tokens_approx(self) -> int:
        return sum(sum(len(m.get("content", "")) for m in e.messages) // 4 for e in self.entries)

    @property
    def experiment_ids(self) -> list[str]:
        return list({e.experiment_id for e in self.entries if e.experiment_id})


@dataclass
class Scenario:
    """A benchmark scenario containing one or more tasks.

    This is the top-level data structure for replay. A scenario like
    "markdown-note-app" or "cpp-coding-glm5" bundles multiple tasks,
    each with its own recording.
    """

    name: str = ""
    description: str = ""
    path: str = ""
    tasks: list[Task] = field(default_factory=list)

    @property
    def all_entries(self) -> list[RecordingEntry]:
        """Flat list of all entries across all tasks (for backward compat)."""
        return [e for t in self.tasks for e in t.entries]

    @property
    def experiment_ids(self) -> list[str]:
        return list({eid for t in self.tasks for eid in t.experiment_ids})

    @property
    def total_requests(self) -> int:
        return sum(t.total_requests for t in self.tasks)

    @property
    def total_tokens_approx(self) -> int:
        return sum(t.total_tokens_approx for t in self.tasks)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "tasks": len(self.tasks),
            "experiments": len(self.experiment_ids),
            "requests": self.total_requests,
            "approx_tokens": self.total_tokens_approx,
        }


def _parse_entry(data: dict) -> RecordingEntry:
    return RecordingEntry(
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


def _load_jsonl(path: Path) -> list[RecordingEntry]:
    entries: list[RecordingEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(_parse_entry(json.loads(line)))
    return entries


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a directory (with manifest) or a single JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario not found: {p}")

    if p.is_dir():
        return _load_scenario_dir(p)

    return _load_scenario_jsonl(p)


def _load_scenario_dir(directory: Path) -> Scenario:
    """Load a scenario from a directory with a scenario.json manifest."""
    manifest_path = directory / "scenario.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario directory {directory} missing scenario.json manifest")

    with open(manifest_path) as f:
        manifest = json.load(f)

    tasks: list[Task] = []
    for task_def in manifest.get("tasks", []):
        recording_path = directory / task_def["recording"]
        if not recording_path.exists():
            raise FileNotFoundError(f"Task recording not found: {recording_path}")
        entries = _load_jsonl(recording_path)
        tasks.append(
            Task(
                id=task_def.get("id", recording_path.stem),
                name=task_def.get("name", recording_path.stem),
                entries=entries,
            )
        )

    return Scenario(
        name=manifest.get("name", directory.name),
        description=manifest.get("description", ""),
        path=str(directory),
        tasks=tasks,
    )


def _load_scenario_jsonl(path: Path) -> Scenario:
    """Load a single JSONL as a one-task scenario (backward compat)."""
    entries = _load_jsonl(path)
    task = Task(
        id="default",
        name=path.stem,
        entries=entries,
    )
    return Scenario(
        name=path.stem,
        path=str(path),
        tasks=[task],
    )


def list_builtin_scenarios() -> list[dict]:
    """List all built-in scenarios shipped with the package."""
    if not BUILTIN_DIR.exists():
        return []

    scenarios: list[dict] = []

    for item in sorted(BUILTIN_DIR.iterdir()):
        if item.is_dir() and (item / "scenario.json").exists():
            try:
                s = load_scenario(item)
                scenarios.append(s.summary())
            except Exception:
                scenarios.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "error": "Failed to load",
                    }
                )
        elif item.suffix == ".jsonl":
            try:
                s = load_scenario(item)
                scenarios.append(s.summary())
            except Exception:
                scenarios.append(
                    {
                        "name": item.stem,
                        "path": str(item),
                        "error": "Failed to load",
                    }
                )

    return scenarios


def get_scenario(name_or_path: str) -> Scenario:
    """Load a scenario by name (built-in) or file/directory path."""
    p = Path(name_or_path)
    if p.exists():
        return load_scenario(p)

    builtin_dir = BUILTIN_DIR / name_or_path
    if builtin_dir.is_dir() and (builtin_dir / "scenario.json").exists():
        return load_scenario(builtin_dir)

    builtin_jsonl = BUILTIN_DIR / f"{name_or_path}.jsonl"
    if builtin_jsonl.exists():
        return load_scenario(builtin_jsonl)

    raise FileNotFoundError(
        f"Scenario '{name_or_path}' not found. "
        f"Provide a path to a .jsonl file, a scenario directory, "
        f"or a built-in scenario name."
    )
