"""Tests for configuration management."""

from agentic_coding_bench.config import (
    BenchmarkConfig,
    build_config,
)


def test_default_config():
    cfg = BenchmarkConfig()
    assert cfg.defeat_cache is True
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
    assert users_seen == {1, 8}
    assert profiles_seen == {"fresh"}
    assert len(scenarios) == 2


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
