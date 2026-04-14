# Changelog

All notable changes to AgenticSwarmBench are documented here.

## [1.0.0] - 2026-04-13

### Added

- Initial release of AgenticSwarmBench - the open-source benchmark for LLM inference under agentic swarm workloads.
- **Five CLI modes:** `speed`, `eval`, `agent`, `record`, and `replay`.
- **110 agentic swarm tasks** across 5 difficulty tiers (trivial → expert) and 5 languages (Python, TypeScript, Rust, Go, SQL).
- **Seven context profiles:** `fresh` (6K), `short` (20K), `medium` (40K), `long` (70K), `full` (100K), `xl` (200K), `xxl` (400K) - simulating real coding sessions from first prompt to deep multi-file debugging.
- **`--model-context-length` flag** - automatically skips profiles that exceed the model's context window.
- **Prefix cache defeat** with unique per-request salt, plus `--cache-mode both` to measure exact cache speedup.
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
