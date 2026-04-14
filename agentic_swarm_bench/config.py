"""Configuration management for agentic-swarm-bench.

Merges CLI arguments, environment variables, and optional YAML config.
CLI > env vars > YAML > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONTEXT_PROFILES = {
    "fresh": 6_000,
    "short": 20_000,
    "medium": 40_000,
    "long": 70_000,
    "full": 100_000,
    "xl": 200_000,
    "xxl": 400_000,
}

# Default sweep when --context-profile realistic (or no profile specified).
# xl/xxl are opt-in via --context-profile or --model-context-length.
REALISTIC_PROFILE_SEQUENCE = ["fresh", "short", "medium", "long", "full"]

SUITE_CONFIGS = {
    "quick": {
        "users": [1, 4, 8],
        "profiles": ["fresh"],
    },
    "standard": {
        "users": [1, 8, 16, 32],
        "profiles": ["medium", "long"],
    },
    "full": {
        "users": [1, 4, 8, 16, 32, 64],
        "profiles": REALISTIC_PROFILE_SEQUENCE,
    },
}


@dataclass
class BenchmarkConfig:
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    api_key_header: str = "Authorization"

    task_range: Optional[str] = None
    tier: Optional[str] = None
    tags: Optional[list[str]] = None

    context_profile: Optional[str] = None
    context_tokens: Optional[int] = None
    model_context_length: Optional[int] = None
    random_context: bool = False

    users: int = 1
    max_users: Optional[int] = None
    suite: Optional[str] = None

    max_output_tokens: int = 512
    repetitions: int = 1
    defeat_cache: bool = True
    cache_mode: str = "cold"  # cold | warm | both
    timeout: float = 300.0

    output: Optional[str] = None
    output_format: str = "markdown"
    dry_run: bool = False

    agent: str = "claude-code"
    proxy_port: int = 19000

    verbose: bool = False

    validate: str = "syntax"

    @property
    def resolved_scenarios(self) -> list[tuple[int, str, int]]:
        """Build the full list of (num_users, profile_name, token_count) to run.

        This is the single source of truth for what scenarios get executed.
        Keeps profiles and token counts always in sync.

        Scenarios exceeding model_context_length are skipped.
        """
        users = self._resolve_users()
        profile_tokens = self._resolve_profile_tokens()

        if self.model_context_length is not None:
            profile_tokens = [
                (p, t) for p, t in profile_tokens
                if t <= self.model_context_length
            ]

        return [(u, profile, tokens) for profile, tokens in profile_tokens for u in users]

    def _resolve_users(self) -> list[int]:
        if self.suite and self.suite in SUITE_CONFIGS:
            user_list = SUITE_CONFIGS[self.suite]["users"]
        else:
            user_list = [self.users]

        if self.max_users is not None:
            user_list = [u for u in user_list if u <= self.max_users]

        return user_list

    def _resolve_profile_tokens(self) -> list[tuple[str, int]]:
        """Return list of (profile_name, token_count) pairs."""
        if self.context_tokens is not None:
            label = f"{self.context_tokens // 1000}K"
            return [(label, self.context_tokens)]

        if self.suite and self.suite in SUITE_CONFIGS:
            profiles = SUITE_CONFIGS[self.suite]["profiles"]
            return [(p, CONTEXT_PROFILES[p]) for p in profiles]

        if self.context_profile and self.context_profile != "realistic":
            p = self.context_profile
            return [(p, CONTEXT_PROFILES[p])]

        return [(p, CONTEXT_PROFILES[p]) for p in REALISTIC_PROFILE_SEQUENCE]

    @classmethod
    def from_env(cls) -> BenchmarkConfig:
        return cls(
            endpoint=os.getenv("ASB_ENDPOINT", ""),
            model=os.getenv("ASB_MODEL", ""),
            api_key=os.getenv("ASB_API_KEY", ""),
            context_tokens=_int_or_none(os.getenv("ASB_CONTEXT_TOKENS")),
            context_profile=os.getenv("ASB_CONTEXT_PROFILE"),
            model_context_length=_int_or_none(os.getenv("ASB_MODEL_CONTEXT_LENGTH")),
            defeat_cache=(os.getenv("ASB_DEFEAT_CACHE", "true").lower() == "true"),
        )

    def merge(self, **overrides) -> BenchmarkConfig:
        """Return a new config with non-None overrides applied."""
        updates = {k: v for k, v in overrides.items() if v is not None}
        current = {k: getattr(self, k) for k in self.__dataclass_fields__}
        current.update(updates)
        return BenchmarkConfig(**current)


def resolve_endpoint(endpoint: str) -> str:
    """Normalize the endpoint URL.

    If the URL already contains /v1/chat/completions, use it as-is.
    Otherwise, append /v1/chat/completions.
    """
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _int_or_none(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_yaml_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        with open(p) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def build_config(
    cli_args: Optional[dict] = None,
    config_file: Optional[str] = None,
) -> BenchmarkConfig:
    """Build final config: CLI > env > YAML > defaults."""
    cfg = BenchmarkConfig()

    if config_file:
        yaml_data = load_yaml_config(config_file)
        cfg = cfg.merge(**yaml_data)

    env_cfg = BenchmarkConfig.from_env()
    env_overrides = {}
    for k in cfg.__dataclass_fields__:
        env_val = getattr(env_cfg, k)
        default_val = getattr(BenchmarkConfig(), k)
        if env_val != default_val:
            env_overrides[k] = env_val
    cfg = cfg.merge(**env_overrides)

    if cli_args:
        cfg = cfg.merge(**cli_args)

    return cfg
