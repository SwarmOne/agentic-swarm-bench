# Changelog

All notable changes to AgenticSwarmBench are documented here.

## [3.4.0] - 2026-04-19

### Summary

Ships two built-in scenarios (`trivial-qa` and `js-coding-opus`) so users can benchmark immediately without recording anything first. Also adds native Anthropic Messages API support to the replay command, a `--task` filter for replaying individual tasks, `--json` stdout mode for piping into other tools, and `--verbose` live progress output.

### Added

- **Built-in `trivial-qa` scenario.** Five trivial single-turn Q&A tasks (~20 tokens each) — a 30-second smoke test that verifies endpoint connectivity and reports baseline TTFT and tok/s.
- **Built-in `js-coding-opus` scenario.** Five multi-turn JavaScript coding sessions (REST API, CSV parser CLI, markdown renderer, state machine, WebSocket chat) recorded with Claude Opus 4.6. Each task has 4 turns with context growing from ~1K to ~40K chars.
- **`--scenario` short flag `-s`.** Replay now accepts `-s` as a shorthand for `--scenario` (the legacy `-w` alias is preserved).
- **`--task` / `-t` filter on `asb replay`.** Replay a single task from a multi-task scenario by ID instead of the entire scenario.
- **`--json` flag on `asb replay`.** Writes JSON results to stdout (human-readable output moves to stderr) for piping into `jq`, CI scripts, or other tools.
- **`--upstream-api` flag on `asb replay`.** Explicitly select `openai` or `anthropic` API format. Auto-detected from URL when omitted (api.anthropic.com → anthropic).
- **Native Anthropic Messages API support.** Replay can now send requests directly to Anthropic endpoints, translating recorded OpenAI-format conversations to the Anthropic Messages API format on the fly.
- **`--verbose` / `-V` flag on `asb replay`.** Shows live per-task progress with phase, request count, and decode tok/s.
- **`asb list-scenarios` shows built-in scenarios.** Registry now discovers and lists all built-in scenarios shipped with the package.
- **Quick smoke test in README.** New "Quick smoke test (30 seconds)" section right after installation with copy-paste commands.

### Changed

- **Replay help text reorganized.** Documented input formats (scenario.json, directory, .jsonl, built-in name), output modes (default, --json, -o, --json -o), and upstream API modes in structured `\b` blocks.
- **README updated.** New "Built-in Scenarios" section with table, CLI examples, and scenario descriptions. All `-w` flags in examples updated to `--scenario`. Architecture tree updated with `trivial-qa/` and `js-coding-opus/` entries.

### Removed

- **`recordings/` directory and `markdown-note-app` scenario.** Replaced by the new built-in scenario system under `scenarios/data/`.

---

## [3.3.0] - 2026-04-17

### Summary

All default output - CLI tables, markdown reports, and JSON summaries - now surfaces **both decode tok/s and prefill tok/s** with explicit labels. Previously only one throughput number was shown (generically labeled "Tok/s"), making it unclear whether the metric measured decode-phase streaming speed or prefill-phase input processing rate. Both are now reported side-by-side everywhere.

### Added

- **Prefill tok/s in CLI output.** Replay bucket stats, replay summary table, speed per-scenario stats, and speed summary table now include prefill tok/s alongside decode tok/s.
- **Prefill tok/s in markdown reports.** The Results summary table has a new "Prefill tok/s" column. The verdict line includes prefill speed when available.
- **`prefill_tok_per_sec` in JSON summary.** The top-level `summary` blob in saved JSON results now includes `prefill_tok_per_sec` percentiles (p50/p95/p99/min/max/mean), so CI consumers get both throughput metrics without parsing per-request data.
- **Explicit decode/prefill labels in methodology.** The report methodology section now distinguishes "Decode tok/s" (output token throughput after first token) from "Prefill tok/s" (input token processing rate during TTFT).

### Changed

- **"Tok/s" renamed to "Decode tok/s" everywhere.** CLI tables, markdown report columns, comparison reports, context scaling charts, concurrency scaling tables, and the agent proxy summary all use the explicit "Decode" label so users know exactly which phase the number measures.

---

## [3.2.0] - 2026-04-17

### Summary

Makes the work-queue scheduling model explicit everywhere and plugs the two randomization holes (agent mode had none; replay's `random` policy wasn't reproducible). Both `asb replay` and `asb agent` now share one scheduler primitive: a pool of J long-lived workers that each pull the next head of a pre-ordered list of `T × R` schedule-tasks. See [docs/SCHEDULING.md](docs/SCHEDULING.md) for the full model and worked example.

### Added

- **`--seed` flag on `asb replay` and `asb agent`.** Pass an integer to make `--policy random` reproducible across runs; the same seed yields the same shuffled order. Omit it for system entropy. When set, the seed is echoed in the dry-run and schedule summary output so it's visible in logs.
- **Full scheduling controls on `asb agent`.** Previously the agent command had no scheduling surface; tasks always ran sequentially, once each, one-at-a-time. Now accepts `--repetitions`, `--max-concurrent`, `--policy`, `--seed`, `--timeout` - mirroring the `replay` shape.
- **`run_work_queue` primitive** in `agentic_swarm_bench.scenarios.schedule`. One function that both replay and agent dispatch through: literal pool of J async workers pulling from the head of a `collections.deque` until drained. Replaces the `asyncio.gather + Semaphore(J)` pattern which was semantically correct but read nothing like the work-queue model we were describing in the docs.
- **`Schedule.seed` field.** Bundled into the dataclass so callers don't need to thread a separate seed parameter through every helper.
- **`docs/SCHEDULING.md`** - authoritative reference for the schedule-task concept, the three orderings (sequential, round_robin, random), the pool-of-J dispatcher, and a worked example. CLI docstrings link here.
- **Tests:** 13 new tests covering `run_work_queue` order preservation, concurrency bound, slot_id assignment, drain-despite-slow-item, J > queue_length, schedule-task round-trip, and seed reproducibility (both via argument and `Schedule.seed`).

### Changed

- **Agent mode defaults to `--policy random`.** A `sequential` default let server-side prefix caches get a free ride whenever two back-to-back invocations of the same agent shared tool blocks / system prompt. Random kills that by default; pass `--policy sequential` explicitly if you want the old behavior.
- **Agent runner is now fully async.** `subprocess.run` (blocking) was replaced with `asyncio.create_subprocess_exec` so N parallel agents actually run in parallel instead of serializing through the event loop. `_start_proxy` / `_stop_proxy` follow suit.
- **Replay dispatch uses one long-lived `httpx.AsyncClient` per slot** instead of a fresh client per schedule-task. This matches what a real user session looks like (one keepalive connection reused across many calls) and removes the TLS-handshake-per-task noise that was contaminating TTFT numbers for small schedule-tasks.
- **Metric field clarification.** `RequestMetrics.user_id` now also carries the slot_id in replay mode (same field, refined meaning); a read-only `slot_id` property was added as an alias. JSON wire format is unchanged for backcompat.

### Fixed

- **Replay `--policy random` was non-reproducible.** `build_execution_queue` was called without a seed, so shuffling used system entropy and two runs with `-r 5 --policy random` would hit tasks in different orders. Pass `--seed N` now for byte-for-byte reproducible dispatch order.
- **Agent mode always ran tasks in the same order.** `get_tasks()` returns `tasks.json` order, and the old runner iterated that list directly. Two `asb agent` invocations would replay the exact same task sequence, letting cache hit-rate inflate silently across repeat runs. Now surfaced and controllable via `--policy`.

### Migration

| Old                                    | New                                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------- |
| `asb agent -t p1-p10`                  | `asb agent -t p1-p10` (now random by default; add `--policy sequential` for old behavior) |
| `asb agent -t p1-p10` (rerun for reps) | `asb agent -t p1-p10 -r 4 --max-concurrent 4`                                             |
| `asb replay … --policy random`         | `asb replay … --policy random --seed N` (for reproducibility)                             |

---

## [3.1.0] - 2026-04-17

### Removed (breaking)

- **`asb replay --users / -u` removed.** The flag was a cache free-ride bug: all N "users" sent byte-identical poisoned payloads concurrently via `asyncio.gather`, so only user 0 did a real cold prefill and users 1..N-1 rode the KV cache for free. This inflated the measured cache hit-rate and made the "N independent users on different projects" story a fiction. The poison mask was seeded by `(task_id, execution_index)` only, with no per-user variation.

### Migration

Use `--repetitions N --max-concurrent N` instead. Each repetition gets a distinct poison seed (`f"{task_id}-exec-{execution_index}"`), so N repetitions produce N genuinely different payloads; `--max-concurrent` caps how many run in parallel. Shared LCP (system prompt) remains cache-eligible across repetitions, matching the realistic "multiple devs, different projects, same system prompt" scenario.

| Old                        | New                                                  |
| -------------------------- | ---------------------------------------------------- |
| `asb replay ... --users 8` | `asb replay ... --repetitions 8 --max-concurrent 8`  |
| `asb replay ... -u 4 -r 3` | `asb replay ... --repetitions 12 --max-concurrent 4` |

Passing `--users` to `asb replay` now exits non-zero with a precise migration hint. `asb speed --users` is **unchanged**: that's a synthetic stress test where N identical clients is the intended stimulus.

### Added

- New guard tests in `tests/test_player.py` asserting two repetitions produce different serialized payloads _and_ preserve the shared LCP byte-for-byte. These lock in the invariant that `--repetitions` is a correct replacement for the removed `--users`.

---

## [3.0.0] - 2026-04-16

One-day follow-up to 2.0.0 polishing the CLI surface from launch feedback. Breaking flag renames with a clean 1:1 migration path.

### Changed (breaking)

- **Cache mode naming:** `--cache-mode cold/warm/both` (on `asb speed`) renamed to `allcold/allwarm/realistic`. `realistic` runs both passes (allcold then allwarm) to measure exact cache speedup. (#4)
- **Replay cache mode:** `asb replay --poison/--no-poison` replaced by `--cache-mode [realistic|allcold|allwarm]`. `realistic` is now the **default** - shared prefix is preserved for KV caching, unique user context is poisoned. Use `--cache-mode allwarm` to send requests as recorded (no poisoning). (#4)

### Added

- **Default CLI subcommand:** bare `asb -e URL -m MODEL -w scenario` now dispatches to `replay` without typing the subcommand. Explicit subcommands are unchanged. (#5)
- **Backwards-compat aliases:** old YAML configs (`cache_mode: cold/warm/both`) still resolve to the new mode names instead of silently falling through. (#4)
- **Comprehensive test suite:** 409 tests (up from 266) covering CLI, eval/direct/tier-3 runners, proxy server, recorder, config env resolution, tool-use/tool-result translation, thinking-token roundtrip, and report generation. (#3)
- **Record/replay as headline feature:** README leads with record → replay quick-start; mode comparison table reordered. (#6)

### Fixed

- `--format json` now emits valid JSON. Rich's `console.print()` was wrapping long lines and injecting bare newlines into the output; switched to stdlib `print()`. (#2)
- `asb compare` "Tied" count no longer inflates from scenarios where both sides have zero successes. These are surfaced as `N excluded: zero completions` in the comparison report. (#2)
- `ASB_ENDPOINT` / `ASB_MODEL` environment variables are now respected. Click's `required=True` was firing before env/YAML values could merge; added `_require_endpoint_model()` helper that runs after `build_config`. (#2)
- `asb eval -t <filter>` that matches no tasks now raises `UsageError` instead of silently running the full 110-task suite. (#2)
- Deduplicated `_detect_upstream_api` that was copied verbatim across `proxy/server.py` and `scenarios/recorder.py`; both now import from `proxy/utils.py`. (#3)

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
