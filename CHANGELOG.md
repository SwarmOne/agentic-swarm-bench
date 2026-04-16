# Changelog

All notable changes to AgenticSwarmBench are documented here.

## [Unreleased]

### Changed

- **Cache mode naming:** `--cache-mode cold/warm/both` (on `asb speed`) renamed to `allcold/allwarm/realistic`. `realistic` runs both passes (allcold then allwarm) to measure exact cache speedup.
- **Replay cache mode:** `asb replay --poison/--no-poison` replaced by `--cache-mode [realistic|allcold|allwarm]`. `realistic` is now the **default** - shared prefix is preserved for KV caching, unique user context is poisoned. Use `--cache-mode allwarm` to send requests as recorded (no poisoning).

### Migration

| Old                           | New                                     |
| ----------------------------- | --------------------------------------- |
| `asb speed --cache-mode cold` | `asb speed --cache-mode allcold`        |
| `asb speed --cache-mode warm` | `asb speed --cache-mode allwarm`        |
| `asb speed --cache-mode both` | `asb speed --cache-mode realistic`      |
| `asb replay --poison`         | `asb replay` (realistic is now default) |
| `asb replay` (no flag)        | `asb replay --cache-mode allwarm`       |

---

## [2.0.0] - 2026-04-15

### Changed

- **Scenarios instead of workloads:** recording/replay/listing now live under `agentic_swarm_bench.scenarios` (recorder, player, registry, schedule, poison) with built-in scenario data shipped in `scenarios/data/`.
- **CLI:** `asb list-workloads` is now **`asb list-scenarios`**. Help text and docs use scenario terminology throughout.
- **Saved runs / reports:** speed-benchmark results use a **`scenarios`** list on the run object and in JSON exports (replacing the old workloads-oriented shape where applicable).

### Removed

- **`agentic_swarm_bench.workloads`** and the **`asb list-workloads`** command. Migrate imports to `agentic_swarm_bench.scenarios` and update any automation that parsed workload-specific fields or paths.

## [1.0.0] - 2026-04-13

### Added

- Initial release of AgenticSwarmBench - the open-source benchmark for LLM inference under agentic swarm workloads.
- **Five CLI modes:** `speed`, `eval`, `agent`, `record`, and `replay`.
- **110 agentic swarm tasks** across 5 difficulty tiers (trivial → expert) and 5 languages (Python, TypeScript, Rust, Go, SQL).
- **Seven context profiles:** `fresh` (6K), `short` (20K), `medium` (40K), `long` (70K), `full` (100K), `xl` (200K), `xxl` (400K) - simulating real coding sessions from first prompt to deep multi-file debugging.
- **`--model-context-length` flag** - automatically skips profiles that exceed the model's context window.
- **Prefix cache defeat** with unique per-request salt, plus `--cache-mode realistic` to measure exact cache speedup (runs allcold + allwarm).
- **Reasoning token detection** for DeepSeek R1, o3, and Claude Extended Thinking - reports thinking overhead vs visible output latency.
- **Workload recording** (`asb record`) - recording proxy that captures real coding sessions as replayable JSONL workload files.
- **Workload replay** (`asb replay`) - replay recorded sessions against any endpoint with context-size grouping and full metrics.
- **`asb list-workloads`** - browse available built-in workloads.
- **Rich reports** - markdown reports with verdict (GOOD / MARGINAL / POOR), key findings, context scaling ASCII charts, concurrency scaling tables, per-profile breakdowns, and performance grade indicators.
- **`asb compare`** - head-to-head comparison reports with delta indicators and winner summary.
- **Agent mode proxy** - Anthropic ↔ OpenAI translation proxy for benchmarking real Claude Code sessions.
- **Claude Code optimization skill** (`skill/SKILL.md`) - natural language skill that runs the benchmark in a loop and auto-tunes serving parameters.
- **Configuration** - CLI args > environment variables > YAML config > defaults.
- **Docker and docker-compose** support.
