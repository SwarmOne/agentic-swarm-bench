"""Tests for configuration management and endpoint resolution."""

import pytest

from agentic_swarm_bench.config import (
    BenchmarkConfig,
    build_config,
    resolve_endpoint,
)


def test_default_config():
    cfg = BenchmarkConfig()
    assert cfg.timeout == 300.0
    assert cfg.max_output_tokens == 512


def test_resolved_scenarios_explicit_tokens():
    cfg = BenchmarkConfig(context_tokens=50000)
    scenarios = cfg.resolved_scenarios
    assert len(scenarios) == 1
    assert scenarios[0] == (1, "50K", 50000)


def test_resolved_scenarios_profile():
    cfg = BenchmarkConfig(context_profile="long")
    scenarios = cfg.resolved_scenarios
    assert len(scenarios) == 1
    assert scenarios[0] == (1, "long", 70000)


def test_resolved_scenarios_realistic_default():
    cfg = BenchmarkConfig()
    scenarios = cfg.resolved_scenarios
    assert len(scenarios) == 5
    profiles = [s[1] for s in scenarios]
    assert profiles == ["fresh", "short", "medium", "long", "full"]
    tokens = [s[2] for s in scenarios]
    assert tokens[0] == 6000
    assert tokens[-1] == 100000


def test_resolved_scenarios_suite_quick():
    cfg = BenchmarkConfig(suite="quick")
    scenarios = cfg.resolved_scenarios
    users_seen = {s[0] for s in scenarios}
    profiles_seen = {s[1] for s in scenarios}
    assert users_seen == {1, 4, 8}
    assert profiles_seen == {"fresh"}
    assert len(scenarios) == 3


def test_resolved_scenarios_suite_standard_mapping():
    """Suite 'standard' should map medium->40K and long->70K, not 6K/20K."""
    cfg = BenchmarkConfig(suite="standard")
    scenarios = cfg.resolved_scenarios
    token_by_profile = {s[1]: s[2] for s in scenarios}
    assert token_by_profile["medium"] == 40000
    assert token_by_profile["long"] == 70000


def test_resolved_scenarios_suite_full():
    cfg = BenchmarkConfig(suite="full")
    scenarios = cfg.resolved_scenarios
    users_seen = sorted({s[0] for s in scenarios})
    assert users_seen == [1, 4, 8, 16, 32, 64]
    assert len(scenarios) == 30  # 5 profiles x 6 user counts


def test_merge_overrides():
    cfg = BenchmarkConfig(model="base")
    cfg2 = cfg.merge(model="overridden", endpoint="http://new")
    assert cfg2.model == "overridden"
    assert cfg2.endpoint == "http://new"
    assert cfg.model == "base"


def test_merge_ignores_none():
    cfg = BenchmarkConfig(model="base")
    cfg2 = cfg.merge(model=None, endpoint=None)
    assert cfg2.model == "base"


def test_build_config_cli_overrides():
    cfg = build_config(cli_args={"endpoint": "http://test", "model": "m"})
    assert cfg.endpoint == "http://test"
    assert cfg.model == "m"


def test_build_config_yaml(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text("endpoint: http://yaml-server:8000\nmodel: yaml-model\nsuite: standard\n")
    cfg = build_config(config_file=str(yaml_file))
    assert cfg.endpoint == "http://yaml-server:8000"
    assert cfg.model == "yaml-model"
    assert cfg.suite == "standard"


def test_build_config_cli_overrides_yaml(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text("model: yaml-model\n")
    cfg = build_config(
        config_file=str(yaml_file),
        cli_args={"model": "cli-model"},
    )
    assert cfg.model == "cli-model"


# ---------------------------------------------------------------------------
# resolve_endpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    """Endpoint resolution: normalize URLs to /chat/completions."""

    def test_bare_url_appends_v1_chat_completions(self):
        assert resolve_endpoint("http://localhost:8000") == (
            "http://localhost:8000/v1/chat/completions"
        )

    def test_url_ending_with_v1_appends_chat_completions(self):
        assert resolve_endpoint("http://localhost:8000/v1") == (
            "http://localhost:8000/v1/chat/completions"
        )

    def test_url_already_has_v1_chat_completions(self):
        url = "http://localhost:8000/v1/chat/completions"
        assert resolve_endpoint(url) == url

    def test_gemini_endpoint_not_doubled(self):
        """The Gemini fix: URL already ends with /chat/completions (no /v1 prefix).
        Must NOT append /v1/chat/completions again."""
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert resolve_endpoint(url) == url

    def test_custom_path_ending_with_chat_completions(self):
        url = "https://myproxy.example.com/custom/path/chat/completions"
        assert resolve_endpoint(url) == url

    def test_trailing_slash_stripped(self):
        assert resolve_endpoint("http://localhost:8000/") == (
            "http://localhost:8000/v1/chat/completions"
        )

    def test_v1_trailing_slash_stripped(self):
        assert resolve_endpoint("http://localhost:8000/v1/") == (
            "http://localhost:8000/v1/chat/completions"
        )

    def test_openai_api(self):
        assert resolve_endpoint("https://api.openai.com") == (
            "https://api.openai.com/v1/chat/completions"
        )

    def test_openai_api_v1(self):
        assert resolve_endpoint("https://api.openai.com/v1") == (
            "https://api.openai.com/v1/chat/completions"
        )


# ---------------------------------------------------------------------------
# _int_or_none
# ---------------------------------------------------------------------------


class TestIntOrNone:
    def test_none_returns_none(self):
        from agentic_swarm_bench.config import _int_or_none

        assert _int_or_none(None) is None

    def test_valid_int_string(self):
        from agentic_swarm_bench.config import _int_or_none

        assert _int_or_none("42") == 42

    def test_invalid_string_returns_none(self):
        from agentic_swarm_bench.config import _int_or_none

        assert _int_or_none("not-a-number") is None

    def test_float_string_returns_none(self):
        from agentic_swarm_bench.config import _int_or_none

        assert _int_or_none("3.14") is None

    def test_zero_string(self):
        from agentic_swarm_bench.config import _int_or_none

        assert _int_or_none("0") == 0


# ---------------------------------------------------------------------------
# BenchmarkConfig.from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_reads_endpoint_and_model(self, monkeypatch):
        monkeypatch.setenv("ASB_ENDPOINT", "http://env-server:8000")
        monkeypatch.setenv("ASB_MODEL", "env-model")
        cfg = BenchmarkConfig.from_env()
        assert cfg.endpoint == "http://env-server:8000"
        assert cfg.model == "env-model"

    def test_reads_api_key(self, monkeypatch):
        monkeypatch.setenv("ASB_API_KEY", "sk-env-key")
        cfg = BenchmarkConfig.from_env()
        assert cfg.api_key == "sk-env-key"

    def test_reads_context_tokens(self, monkeypatch):
        monkeypatch.setenv("ASB_CONTEXT_TOKENS", "50000")
        cfg = BenchmarkConfig.from_env()
        assert cfg.context_tokens == 50000

    def test_reads_model_context_length(self, monkeypatch):
        monkeypatch.setenv("ASB_MODEL_CONTEXT_LENGTH", "8192")
        cfg = BenchmarkConfig.from_env()
        assert cfg.model_context_length == 8192

    def test_invalid_context_tokens_becomes_none(self, monkeypatch):
        monkeypatch.setenv("ASB_CONTEXT_TOKENS", "bad")
        cfg = BenchmarkConfig.from_env()
        assert cfg.context_tokens is None

    def test_missing_vars_give_defaults(self, monkeypatch):
        for var in ("ASB_ENDPOINT", "ASB_MODEL", "ASB_API_KEY",
                    "ASB_CONTEXT_TOKENS", "ASB_MODEL_CONTEXT_LENGTH"):
            monkeypatch.delenv(var, raising=False)
        cfg = BenchmarkConfig.from_env()
        assert cfg.endpoint == ""
        assert cfg.model == ""


# ---------------------------------------------------------------------------
# build_config: env var priority
# ---------------------------------------------------------------------------


def test_build_config_env_overrides_yaml(tmp_path, monkeypatch):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text("model: yaml-model\n")
    monkeypatch.setenv("ASB_MODEL", "env-model")
    cfg = build_config(config_file=str(yaml_file))
    assert cfg.model == "env-model"


def test_build_config_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("ASB_MODEL", "env-model")
    cfg = build_config(cli_args={"model": "cli-model"})
    assert cfg.model == "cli-model"


def test_build_config_yaml_unknown_key_raises(tmp_path):
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text("model: test\nunknown_key: value\n")
    with pytest.raises((TypeError, KeyError)):
        build_config(config_file=str(yaml_file))


# ---------------------------------------------------------------------------
# model_context_length filters resolved_scenarios
# ---------------------------------------------------------------------------


def test_model_context_length_filters_scenarios():
    # fresh=6K, short=20K, medium=40K — set limit to 25K; expect only fresh + short
    cfg = BenchmarkConfig(model_context_length=25000)
    scenarios = cfg.resolved_scenarios
    token_counts = [t for _, _, t in scenarios]
    assert all(t <= 25000 for t in token_counts)
    assert any(t <= 20000 for t in token_counts)


def test_model_context_length_all_filtered_empty():
    cfg = BenchmarkConfig(model_context_length=1)
    assert cfg.resolved_scenarios == []
