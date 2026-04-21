<p align="center">
  <img src="https://raw.githubusercontent.com/SwarmOne/agentic-swarm-bench/main/assets/logo.png" alt="AgenticSwarmBench" width="640" />
</p>

<p align="center">
  <strong>The open-source benchmark for LLM inference under agentic scenarios</strong><br>
  Created by <a href="https://swarmone.ai"><img src="https://raw.githubusercontent.com/SwarmOne/agentic-swarm-bench/main/assets/swarmone-logo.svg" alt="SwarmOne" height="20" style="vertical-align: middle;" /></a> - the AI-native cloud for agentic scenarios
</p>

<p align="center">
  <a href="https://pypi.org/project/agentic-swarm-bench/"><img src="https://img.shields.io/pypi/v/agentic-swarm-bench" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/swarmone/agentic-swarm-bench/actions"><img src="https://github.com/swarmone/agentic-swarm-bench/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/agentic-swarm-bench/"><img src="https://img.shields.io/pypi/pyversions/agentic-swarm-bench" alt="Python"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#why-agentic-swarm">Why Agentic Swarm</a> &bull;
  <a href="#built-in-scenarios">Built-in Scenarios</a> &bull;
  <a href="#scenario-recording--replay"><strong>Record & Replay</strong></a> &bull;
  <a href="#benchmark-modes">Modes</a> &bull;
  <a href="#the-110-tasks">Tasks</a> &bull;
  <a href="#context-control">Context Control</a> &bull;
  <a href="#prefix-cache-poisoning">Cache Poisoning</a> &bull;
  <a href="#reports">Reports</a> &bull;
  <a href="#docker">Docker</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/SwarmOne/agentic-swarm-bench/main/assets/demo.gif" alt="AgenticSwarmBench demo" width="720" />
</p>

---

## Why Agentic Swarm?

When Claude Code opens a file, reads 2,000 lines, edits three functions, runs tests, and reads the error output - that's **5+ LLM round-trips with 40-100K token contexts** growing each turn. Every turn adds tool results, file contents, and error traces to the conversation.

**No existing benchmark simulates this.**

- **SWE-bench** measures model quality on GitHub issues. It doesn't measure inference speed.
- **LMSys / Chatbot Arena** measures chatbot throughput at ~2K context. Agentic swarm contexts are 20-80x larger.
- **Generic LLM benchmarks** send uniform requests. Agentic swarm scenarios have system prompts with tool schemas, multi-turn history, code files, and growing context windows.

**AgenticSwarmBench fills that gap** - it benchmarks your LLM serving stack under the exact access patterns that Claude Code, Cursor, Windsurf, and Copilot generate.

| What makes it different        |                                                                                                                                                                                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Record & replay**            | **The headline feature.** Capture real coding sessions as replayable JSONL scenarios, then benchmark them against any endpoint. Your actual traffic, your actual context patterns - no synthetic approximation needed. |
| **Three benchmark modes**      | Record/replay (your real sessions), speed (synthetic load), agent (real multi-turn) - plus reporting and comparison                                                                                                    |
| **Agentic swarm context**      | Pads requests with real-looking agentic sessions - system prompts with tool definitions, prior conversation turns, code files, tool call results, error traces                                                         |
| **Growing context simulation** | Profiles simulate how context grows during a real coding session: fresh (6K) → short (20K) → medium (40K) → long (70K) → full (100K) → xl (200K) → xxl (400K)                                                          |
| **Prefix cache poisoning**     | Pre-processed recordings with varied punctuation and capitalization break prefix caching without adding artificial content, ensuring true cold-start measurements                                                       |
| **Cache impact measurement**   | `--cache-mode realistic` runs allcold + allwarm to show exact prefix cache speedup (10x cost difference on Anthropic)                                                                                                  |
| **Reasoning token detection**  | Automatically detects thinking/reasoning tokens (DeepSeek R1, o3, Claude Extended Thinking) and reports thinking overhead vs visible output latency                                                                    |
| **110 agentic swarm tasks**    | 5 difficulty tiers, 5 languages (Python, TypeScript, Rust, Go, SQL) - from single-function fixes to full-stack refactors                                                                                               |
| **Docker one-liner**           | Point at any vLLM / SGLang / TGI / OpenAI-compatible endpoint and go                                                                                                                                                   |

---

## Quick Start

### Install

```bash
uv pip install agentic-swarm-bench             # with uv (recommended)
pip install agentic-swarm-bench                # or with pip

uv pip install "agentic-swarm-bench[proxy]"    # add proxy support (for agent / record modes)
```

### Quick smoke test (30 seconds)

Verify your endpoint works before doing anything else - replay the built-in `trivial-qa` scenario:

```bash
asb replay -e http://your-server:8000 -m your-model --scenario trivial-qa
```

This fires 5 trivial single-turn requests (~20 tokens each) and reports TTFT and tok/s. If numbers come back, your setup works. Then try the real thing:

```bash
asb replay -e http://your-server:8000 -m your-model --scenario js-coding-opus
```

This replays 5 multi-turn agentic coding sessions (REST API, CLI tool, WebSocket chat, audit trail, search/batch) recorded with Claude Opus 4.6 - growing context from ~1K to ~40K chars across 4 turns each.

### Record a real session, then replay it anywhere

The fastest way to get meaningful numbers: record what you actually do, then replay it.

```bash
# 1. Start the recording proxy
asb record -e http://your-gpu-server:8000 -m your-model

# 2. Point your agent at the proxy (runs on localhost:19000)
ANTHROPIC_BASE_URL=http://localhost:19000 claude

# 3. Do your normal work. Ctrl+C when done. You now have a .jsonl recording.

# 4. Replay that session against any endpoint
asb replay -e http://new-server:8000 -m my-model --scenario my-session.jsonl
```

This captures your real context patterns, real token counts, and real multi-turn behavior - then lets you A/B test endpoints with your actual workload.

### Run a synthetic speed test

If you don't have a recording yet, the speed mode generates realistic agentic context synthetically:

```bash
# Quick speed test - 1 and 8 concurrent agents at fresh (6K) context
asb speed \
  --endpoint http://localhost:8000 \
  --model my-model \
  --suite quick

# Full suite with report - sweeps all context sizes and concurrency levels
asb speed \
  --endpoint http://localhost:8000 \
  --model my-model \
  --suite full \
  --output report.md
```

`asb` is the short alias. `agentic-swarm-bench` also works.

**Endpoint URL:** Pass any URL. If it doesn't end with `/v1/chat/completions`, the path is appended automatically. Both of these work:

```bash
asb speed -e http://localhost:8000 -m my-model
asb speed -e https://api.example.com/v1/chat/completions -m my-model
```

**Authentication:** By default, `--api-key` is sent as `Authorization: Bearer <key>`. If your endpoint uses a different header:

```bash
asb speed -e URL -m MODEL -k MY_KEY --api-key-header X-API-Key
```

**Dry run:** Preview what will be sent without making requests:

```bash
asb speed -e URL -m MODEL --dry-run
```

> **Note:** Some inference endpoints may not return detailed error messages on failure. Use `--dry-run` to validate your configuration before running a full benchmark.

### Docker

```bash
docker run --rm -v $(pwd)/results:/results \
  swarmone/agentic-swarm-bench speed \
  --endpoint http://host.docker.internal:8000 \
  --model my-model \
  --suite quick \
  --output /results/report.md
```

---

## Built-in Scenarios

Two ready-made scenarios ship with the package so you can benchmark immediately - no recording needed:

| Scenario             | Type                  | Tasks | Turns/task | Context          | What it measures                                    |
| -------------------- | --------------------- | ----: | ---------: | ---------------- | --------------------------------------------------- |
| **`trivial-qa`**     | Non-agentic baseline  |     5 |          1 | ~20 tokens each  | Raw single-turn speed (TTFT, tok/s)                 |
| **`js-coding-opus`** | Real agentic sessions |     5 |          4 | ~1K → ~40K chars | Multi-turn agentic performance with growing context |

```bash
# List all built-in scenarios
asb list-scenarios

# Quick smoke test - 5 trivial questions, ~20 tokens each
asb replay -e http://your-server:8000 -m your-model --scenario trivial-qa

# Real agentic workload - 5 JS coding sessions recorded with Claude Opus 4.6
asb replay -e http://your-server:8000 -m your-model --scenario js-coding-opus

# Replay a single task from a scenario
asb replay -e http://your-server:8000 -m your-model --scenario js-coding-opus --task build-rest-api

# Run multiple repetitions for stable numbers
asb replay -e http://your-server:8000 -m your-model --scenario js-coding-opus --repetitions 3
```

**`trivial-qa`** - Five trivial single-turn questions (capital of France, largest planet, boiling point of water, speed of light, binary conversion). Non-agentic baseline with minimal context. Useful as a quick smoke test and for comparing agentic vs non-agentic performance on the same endpoint.

**`js-coding-opus`** - Five independent JavaScript coding sessions (rate limiting middleware, CLI admin tool, WebSocket real-time updates, activity log/audit trail, search & batch operations). Each task has 4 turns of real multi-turn conversation with growing context. Recorded with Claude Opus 4.6 against a TaskFlow API project.

---

## Benchmark Modes

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                             AgenticSwarmBench                                 │
├──────────────────────┬──────────────────┬─────────────────────────────────────┤
│  asb record / replay │  asb speed       │  asb agent                          │
│  ★ recommended       │                  │                                     │
│                      │  Synthetic       │  Runs Claude Code (or any agent)    │
│  Capture YOUR real   │  agentic context │  end-to-end with benchmark tasks    │
│  coding sessions as  │  → endpoint      │  through a metrics proxy            │
│  JSONL, then replay  │                  │                                     │
│  against any endpoint│  1 request per   │  5-15 real requests per task        │
│                      │  measurement     │  with tool use, file I/O,           │
│  Real multi-turn     │                  │  growing context                    │
│  conversations with  │  Measures:       │                                     │
│  your actual context │  TTFT, tok/s     │  Measures:                          │
│                      │  ITL, prefill    │  Multi-turn latency compounding     │
│  Measures:           │                  │  Context growth over a session      │
│  Same metrics, but   │                  │                                     │
│  from YOUR real data │                  │                                     │
└──────────────────────┴──────────────────┴─────────────────────────────────────┘
```

### Scenario Recording & Replay

**This is the most valuable way to benchmark.** Synthetic load tells you what an endpoint _can_ do in theory. Record/replay tells you what it _actually does_ with your traffic. Record a real coding session once, then replay that exact sequence of requests against any endpoint, hardware config, or model - same context, same token counts, same multi-turn patterns.

Why this matters: agentic sessions have a unique shape. Context starts small and grows unpredictably. Some turns are tiny follow-ups; others dump 20K tokens of file contents. Synthetic benchmarks can approximate this, but a recording captures the real thing.

#### `asb record` - Capture a Real Session

Starts a recording proxy between your agent and your LLM endpoint. Every request/response pair is saved as a JSONL recording:

```bash
# Record with an OpenAI-compatible upstream
asb record \
  -e http://your-gpu-server:8000 \
  -m your-model

# Record with Anthropic (auto-detected from URL)
asb record \
  -e https://api.anthropic.com \
  -m claude-sonnet-4-20250514 \
  -k $ANTHROPIC_API_KEY \
  --api-key-header x-api-key \
  -o my-session.jsonl

# Custom output file and port
asb record \
  -e http://your-gpu-server:8000 \
  -m your-model \
  -o my-session.jsonl \
  -P 9000
```

Then point Claude Code at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:19000 claude
```

The recorder supports **two upstream modes**:

- **OpenAI-compatible** (default): translates Anthropic Messages API → OpenAI format before forwarding
- **Anthropic passthrough**: forwards requests natively to Anthropic's API - no translation, full fidelity. Auto-detected when the endpoint is `api.anthropic.com`, or set explicitly with `--upstream-api anthropic`.

Both modes save the recording in OpenAI format for replay. Stop with `Ctrl+C` when done.

#### `asb replay` - Replay Against Any Endpoint

Take a recorded scenario and replay it against a different endpoint, hardware, or configuration:

```bash
# Replay a session against a new endpoint
asb replay \
  -e http://new-server:8000 \
  -m my-model \
  --scenario my-session.jsonl

# Replay a scenario directory with schedule
asb replay \
  -e http://new-server:8000 \
  -m my-model \
  --scenario ./scenarios/my-scenario/ \
  --repetitions 3 --max-concurrent 5 --policy sequential

# Default: realistic cache mode (shared prefix preserved, user context poisoned)
asb replay -e URL -m MODEL --scenario scenario

# Preview without sending requests
asb replay -e URL -m MODEL --scenario session.jsonl --dry-run

# Replay just the beginning of a session (up to 1M cumulative prompt tokens)
asb replay -e URL -m MODEL --scenario session.jsonl --slice-tokens 1000000
```

**Scheduling:** Control how tasks execute with `--repetitions`, `--max-concurrent`, and `--policy` (round_robin, sequential).

**Cache mode:** The default (`--cache-mode realistic`) preserves the shared prefix so it can be KV-cached, but poisons each user's unique context so it doesn't. Use `--cache-mode allwarm` for all-cached (optimistic) numbers or `--cache-mode allcold` to defeat caching entirely. See [Prefix Cache Poisoning](#prefix-cache-poisoning) for how this works.

**History mode:** The default (`--history-mode live`) captures the server's actual responses during streaming and feeds them into the next turn's conversation history. This is essential for correct prefix-cache measurement when replaying against a model different from the one that made the recording - without it, recorded assistant messages from the original model cause KV-cache prefix mismatches on every turn. Use `--history-mode recorded` for the legacy behavior of sending each entry's recorded messages verbatim.

**Slicing scenarios:** Real sessions grow from small contexts to large ones. `--slice-tokens N` replays requests from the start until cumulative prompt tokens reach N.

**Output modes:** `--verbose` (`-V`) shows a Rich live-updating table with per-task progress, phase, request counts, and decode tok/s.

**Early abort:** `--max-consecutive-failures N` stops the entire run if any worker slot hits N consecutive failures (HTTP errors, timeouts). Useful when pointing at an endpoint that may be down or producing garbage:

```bash
asb replay -e URL -m MODEL --scenario js-coding-opus --max-consecutive-failures 5
```


#### `asb list-scenarios` - Browse Built-in Scenarios

```bash
asb list-scenarios
asb list-scenarios --format json
```

### `asb speed` - Inference Speed Under Agentic Load

When you don't have a recording yet, or want to test at specific context sizes and concurrency levels, `asb speed` generates realistic agentic context synthetically. Each request is padded so the model sees what it would see in a real coding session - system prompts with tool schemas, multi-turn conversation history, file contents, and error traces:

````
┌─ system ───────────────────────────────────────────────────────────────────┐
│ "You are an expert software engineer assistant integrated into a code      │
│  editor. You have access to the user's full project codebase..."           │
└────────────────────────────────────────────────────────────────────────────┘
┌─ user ─────────────────────────────────────────────────────────────────────┐
│ <tool name="Read">...</tool>              ← tool definitions (Read, Write, │
│ <tool name="Write">...</tool>               Edit, Bash, Grep, etc.)        │
│                                                                            │
│ <user_turn> Review src/auth/middleware.py  ← synthetic prior conversation  │
│   ```python                                 turns with code files, error   │
│   def handle_request(...)                   traces, and assistant replies  │
│   ```                                       (repeated to fill target       │
│ </user_turn>                                 context size)                 │
│ <assistant_turn> I can see the issue...                                    │
│ </assistant_turn>                                                          │
│                                                                            │
│ ---                                                                        │
│ Based on the codebase above, <task prompt from tasks.json>                 │
└────────────────────────────────────────────────────────────────────────────┘
````

The task prompt (e.g. "Write a Python function that takes a list of integers and returns the largest one") comes from the [110 built-in tasks](#the-110-tasks). The padding around it is what makes this an **agentic** benchmark - it simulates the accumulated context of a real coding session.

```bash
# Default: sweeps context sizes from fresh (6K) → full (100K)
asb speed -e http://localhost:8000 -m my-model

# Specific concurrency (32 concurrent agents) at long context
asb speed -e http://localhost:8000 -m my-model -u 32 -p long

# Fixed token count - stress test at exactly 50K tokens
asb speed -e http://localhost:8000 -m my-model -c 50000 -u 16

# Cap max users - run a full suite but limit concurrency to 16
asb speed -e http://localhost:8000 -m my-model --suite full --max-users 16

# Measure prefix cache impact - runs allcold then allwarm
asb speed -e http://localhost:8000 -m my-model --cache-mode realistic

# JSON-only output (for CI/CD pipelines)
asb speed -e http://localhost:8000 -m my-model --format json -o results.json
```

**Metrics:** TTFT, decode tok/s per user, prefill tok/s, ITL (p50/p95/p99), aggregate throughput, reasoning token overhead. When the endpoint returns `prompt_tokens` in the response, actual token counts are shown alongside estimates.

### `asb agent` - End-to-End Agent Benchmark

The other modes measure individual requests. `asb agent` measures what it **feels like** to use an endpoint - it runs a **real agent process** (Claude Code by default) end-to-end and records timing for every LLM call across the entire multi-turn session.

Here's what a single task run looks like:

```
You run:    asb agent -e http://localhost:8000 -m my-model -t p1-p10

What happens for each task:

  ┌─────────────┐         ┌─────────────────┐         ┌──────────────┐
  │ Claude Code │ ──────► │  ASB proxy      │ ──────► │ Your endpoint│
  │ (real agent)│ ◄────── │  (translates    │ ◄────── │ (vLLM, etc.) │
  │             │         │   Anthropic →   │         │              │
  │ reads files │         │   OpenAI, logs  │         │              │
  │ writes code │         │   per-request   │         │              │
  │ runs tests  │         │   timing)       │         │              │
  │ iterates    │         │                 │         │              │
  └─────────────┘         └─────────────────┘         └──────────────┘

  Turn 1:  Claude reads the task           →  6K context   →  TTFT 200ms
  Turn 2:  Claude reads 3 files            →  25K context  →  TTFT 800ms
  Turn 3:  Claude writes code              →  35K context  →  TTFT 1.2s
  Turn 4:  Claude runs tests, gets errors  →  50K context  →  TTFT 2.1s
  Turn 5:  Claude fixes the code           →  60K context  →  TTFT 3.5s
  Turn 6:  Claude runs tests again         →  70K context  →  TTFT 4.8s
  ...
```

This captures **latency compounding over a real session**. Each turn's context naturally grows because it includes prior turns, file contents, tool outputs, and error traces. The proxy records TTFT, tok/s, and context size for every request.

```bash
asb agent -e http://localhost:8000 -m my-model -t p1-p10

# Use a different agent (any CLI that accepts a prompt)
asb agent -e http://localhost:8000 -m my-model -t p1-p10 --agent-cmd my-agent
```

**Record/Replay vs Speed vs Agent:**

|                                 | `record` / `replay`                      | `speed`                              | `agent`                                        |
| ------------------------------- | ---------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| **What talks to your endpoint** | You during `record`, ASB during `replay` | ASB directly (one synthetic request) | A real agent (Claude Code) through a proxy     |
| **Number of requests per task** | Whatever the real session had            | 1                                    | 5-15+ (real tool-use turns)                    |
| **Context**                     | Your actual session context              | Synthetic padding to target size     | Grows naturally as the agent works             |
| **Use case**                    | **Benchmark with your real traffic**     | Raw throughput at controlled sizes   | "What does it feel like to use this endpoint?" |

### `asb eval` - Code Correctness (experimental)

Optional mode that sends the same tasks with agentic context, but validates the generated code instead of measuring speed. Useful for checking if your model still produces correct code under large-context pressure.

```bash
asb eval -e http://localhost:8000 -m my-model -t p1-p25 -v syntax      # does it parse?
asb eval -e http://localhost:8000 -m my-model -t p1-p25 -v execution   # does it run?
```

### `asb list-tasks` - Browse Available Tasks

```bash
asb list-tasks                        # Show all 110 tasks
asb list-tasks -t trivial             # Filter by tier
asb list-tasks --tags typescript,rust  # Filter by language
asb list-tasks --format json          # JSON output
```

---

## The 110 Tasks

Tasks simulate real agentic coding scenarios across 5 difficulty tiers and 5 languages:

| Tier        | Range     | What it simulates                                                          |
| ----------- | --------- | -------------------------------------------------------------------------- |
| 1 - Trivial | P1-P10    | Quick fixes: rename a variable, add a type hint, write a one-liner         |
| 2 - Easy    | P11-P25   | Single-file tasks: implement a function, write a CLI tool, parse a file    |
| 3 - Medium  | P26-P50   | Multi-function work: build an API endpoint, write tests, refactor a module |
| 4 - Hard    | P51-P75   | Complex tasks: networking, concurrency, database queries, full programs    |
| 5 - Expert  | P76-P100  | Real-world projects: multi-file apps, distributed systems, full-stack      |
| Multi-lang  | P101-P110 | TypeScript, Rust, Go, SQL tasks across all difficulty levels               |

**Languages:** Python (P1-P100), TypeScript, Rust, Go, SQL (P101-P110). Filter with `--tags typescript,rust,go`.

Tasks define **what to generate**. Context size is controlled separately - so you can benchmark a trivial fix inside a massive 100K-token coding session.

---

## Context Control

Context size simulates **where you are in a real coding session**:

| Profile     | Tokens | What it simulates                                                      |
| ----------- | ------ | ---------------------------------------------------------------------- |
| `fresh`     | ~6K    | Just opened the project - system prompt + first question               |
| `short`     | ~20K   | A few turns in - read a couple files, made one edit                    |
| `medium`    | ~40K   | Mid-session - several file reads, tool calls, error traces             |
| `long`      | ~70K   | Deep session - many edits, test runs, debugging cycles                 |
| `full`      | ~100K  | Long session approaching context limit - everything accumulated        |
| `xl`        | ~200K  | Extended session - large codebases, long test output, multi-file edits |
| `xxl`       | ~400K  | Maximum depth - for models with 400K+ context windows                  |
| `realistic` | Mixed  | Sweeps fresh → full (default) - simulates a full session lifecycle     |

Every request is padded with content that looks like a real agentic coding session:

- System prompt with tool schemas (Read, Write, Edit, Bash, Grep, etc.)
- Prior conversation turns with file contents
- Tool call results and error traces
- Growing context that mimics how sessions actually evolve

```bash
# Simulate a deep coding session (70K context)
asb speed -e URL -m MODEL --context-profile long

# Long-context models: test at 200K or 400K
asb speed -e URL -m MODEL --context-profile xl
asb speed -e URL -m MODEL --context-profile xxl

# Exact token count
asb speed -e URL -m MODEL --context-tokens 50000

# Default: sweeps fresh → short → medium → long → full
asb speed -e URL -m MODEL
```

### Model Context Window

Use `--model-context-length` to tell the benchmark your model's maximum context window. Any profiles that exceed it are automatically skipped:

```bash
# Model supports up to 128K - xl (200K) and xxl (400K) are skipped automatically
asb speed -e URL -m MODEL --suite full --model-context-length 128000

# Model supports 400K - run everything including xxl
asb speed -e URL -m MODEL --context-profile xxl --model-context-length 400000
```

This is useful when running suites or `realistic` sweeps against models with different context limits - no need to manually pick a profile.

---

## Prefix Cache Poisoning

LLM inference engines cache the KV state of common prefixes so repeated requests skip prefill. This makes benchmarks look artificially fast - you're measuring cache hits, not real inference.

AgenticSwarmBench defeats the prefix cache using **pre-processed recordings** with varied punctuation and capitalization treatments. Each recording receives a unique transformation that shifts BPE token boundaries, invalidating the KV cache without altering the semantic content the model sees.

This mimics what actually happens in real coding sessions: when an agent edits a file mid-conversation, the context changes from the edit point onward, breaking the cache naturally.

- **`asb replay`**: Each recording has pre-applied cache-defeat treatments. Different repetitions use different recordings to ensure cache invalidation across runs.

### Controlling cache behavior

Both `asb speed` and `asb replay` accept `--cache-mode` with three options:

| Mode        | What it does                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `allcold`   | Every request defeats the KV cache. Measures true cold-start latency. Default for `asb speed`.                                       |
| `allwarm`   | No poisoning - requests arrive as-is and the server can cache freely. Measures best-case latency.                                    |
| `realistic` | Preserves the shared prefix (system prompt) so it can be cached; poisons only the unique per-user portion. Default for `asb replay`. |

```bash
# speed: default is allcold (true cold-start numbers)
asb speed -e URL -m MODEL

# speed: measure both extremes in one run
asb speed -e URL -m MODEL --cache-mode realistic

# speed: best-case cached numbers
asb speed -e URL -m MODEL --cache-mode allwarm

# replay: default is realistic (production-accurate)
asb replay -e URL -m MODEL --scenario scenario

# replay: all-cached (optimistic upper bound)
asb replay -e URL -m MODEL --scenario scenario --cache-mode allwarm
```

`--cache-mode realistic` on `asb speed` runs each scenario twice (allcold then allwarm) and reports both. Anthropic charges 10x less for cached tokens ($0.30 vs $3.00/M), so knowing your cache hit rate matters.

## Reasoning Token Detection

Models like DeepSeek R1, o3, and Claude Extended Thinking produce **thinking tokens** before visible output. AgenticSwarmBench automatically detects `reasoning_content` in the streaming response and reports:

| Metric                | Description                                        |
| --------------------- | -------------------------------------------------- |
| **TTFT (thinking)**   | Time to first reasoning token                      |
| **TTFT (visible)**    | Time to first visible output token                 |
| **Thinking overhead** | Extra latency from reasoning before visible output |
| **Thinking tokens**   | Count of reasoning tokens generated                |

This is critical for agentic swarm scenarios - a reasoning model that takes 5 seconds to "think" before emitting code changes the UX of the entire editing session.

---

## What Good Looks Like

Reference ranges from common setups (your numbers will vary by hardware, model size, and serving stack):

| Setup                                   | Context | Users |   TTFT | Tok/s/user | Notes                               |
| --------------------------------------- | ------- | ----: | -----: | ---------: | ----------------------------------- |
| vLLM on 1x A100 (80GB), 7B model        | 6K      |     1 | ~100ms |    ~80-120 | Baseline: fast model, short context |
| vLLM on 1x A100 (80GB), 7B model        | 40K     |     8 |  ~2-4s |     ~20-40 | Typical agentic scenario            |
| vLLM on 1x A100 (80GB), 7B model        | 100K    |    32 | ~8-15s |      ~5-15 | Stress test                         |
| SGLang on 1x H100, 70B model            | 6K      |     1 | ~200ms |     ~40-60 | Larger model, faster GPU            |
| SGLang on 1x H100, 70B model            | 40K     |     8 |  ~3-6s |     ~10-25 | Agentic sweet spot                  |
| API provider (e.g. Together, Fireworks) | 40K     |     8 |  ~2-8s |     ~15-40 | Varies by provider/load             |

**Rules of thumb for agentic swarm scenarios:**

- **TTFT < 3s** at 40K context → responsive editing experience
- **Tok/s > 30/user** → code appears to stream smoothly
- **TTFT < 10s** at 100K context → acceptable for deep sessions
- **Agg tok/s scales sub-linearly** with users - expect ~60-70% efficiency at 8x concurrency

> **A note on ITL:** Inter-Token Latency measures the gap between SSE `data:` lines as received by the client. Very low values (< 1ms) typically reflect HTTP/TCP buffering, not actual token generation speed. Use tok/s as the primary throughput metric; ITL is best for relative comparisons across scenarios.

---

## Reports

Reports are designed to answer one question fast: **is this endpoint good enough for agentic swarm scenarios?**

Every report includes:

1. **Verdict** - 🟢 GOOD / 🟡 MARGINAL / 🔴 POOR for agentic swarm scenarios, with the key numbers
2. **Key Findings** - auto-generated insights: TTFT scaling ratio, throughput range, concurrency efficiency, thinking overhead
3. **Summary table** - TTFT, tok/s, ITL at each concurrency level and context size, with color-coded grade icons per row
4. **What This Means for Agentic Swarm** - maps raw metrics to user experience ("instant response", "smooth streaming", "sluggish, frustrating")
5. **Context Scaling chart** - ASCII chart showing how TTFT and tok/s change as context grows
6. **Concurrency Scaling** - efficiency percentages at each concurrency level with color grades
7. **Per-profile breakdown** - detailed numbers per context size
8. **Reasoning token analysis** - thinking overhead when using reasoning models
9. **Methodology** - what was measured, how, and what the grade thresholds mean

The CLI also prints a final verdict line after every benchmark:

```
  Verdict: GOOD for agentic swarm scenarios at medium context
```

Compare two runs (includes head-to-head table, ASCII bar chart, and winner summary):

```bash
asb compare --baseline a.json --candidate b.json -o comparison.md
```

---

## Docker

### Build

```bash
docker build -t swarmone/agentic-swarm-bench .
```

### Run

```bash
# Speed benchmark
docker run --rm -v $(pwd)/results:/results \
  swarmone/agentic-swarm-bench speed \
  -e http://host.docker.internal:8000 \
  -m my-model --suite full \
  -o /results/report.md

# Recording proxy for agentic mode
docker-compose up proxy
```

### Docker Compose

```bash
export ASB_ENDPOINT=http://your-gpu-server:8000
export ASB_MODEL=your-model-name

docker-compose run agentic-swarm-bench
```

---

## Configuration

AgenticSwarmBench merges configuration from four sources (highest priority first):

1. **CLI arguments** - `--endpoint`, `--model`, `--context-tokens`, etc.
2. **Environment variables** - `ASB_ENDPOINT`, `ASB_MODEL`, etc.
3. **YAML config file** - `asb --config bench.yml speed ...`
4. **Defaults** - sensible defaults for everything

### YAML Config Example

```yaml
# bench.yml
endpoint: http://my-gpu-server:8000
model: my-model
suite: standard
```

### Environment Variables

| Variable                   | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `ASB_ENDPOINT`             | OpenAI-compatible endpoint URL                      |
| `ASB_MODEL`                | Model name                                          |
| `ASB_API_KEY`              | API key for the endpoint                            |
| `ASB_CONTEXT_TOKENS`       | Default context size in tokens                      |
| `ASB_CONTEXT_PROFILE`      | Default context profile                             |
| `ASB_MODEL_CONTEXT_LENGTH` | Model's max context window - skips larger scenarios |

---

## Architecture

```
agentic-swarm-bench/
  agentic_swarm_bench/
    cli.py              ← Click CLI (asb record | replay | speed | agent | eval | ...)
    config.py           ← Config: CLI > env > YAML > defaults

    scenarios/
      recorder.py       ← Recording proxy: captures real sessions as JSONL recordings
      player.py         ← Replay engine: replays scenarios against any endpoint
      registry.py       ← Load/list/resolve scenarios (file path or built-in name)
      schedule.py       ← Execution schedule: repetitions, concurrency, ordering policy
      poison.py         ← Prefix-cache poisoning hooks (uses pre-processed recordings)
      data/
        trivial-qa/     ← Non-agentic baseline (5 single-turn Q&A tasks, with evaluations)
        js-coding-opus/ ← Agentic JS coding sessions (5 multi-turn tasks)

    tasks/
      tasks.json        ← 110 agentic swarm tasks, P1-P110
      registry.py       ← Load/filter tasks by tier, range, tags, language
      context/
        codebase_context.py ← Agentic session context: tool schemas, file contents, conversation turns

    runner/
      direct.py         ← Speed mode: direct endpoint benchmark with agentic context
      eval_runner.py    ← Eval mode: code correctness validation
      claude_code.py    ← Agent mode: Claude Code orchestration through recording proxy

    proxy/
      server.py         ← Agent-mode proxy (FastAPI) - Anthropic ↔ OpenAI translation
      padding.py        ← Context padding for proxy mode
      translators.py    ← API format translation

    metrics/
      collector.py      ← Per-request metrics: TTFT, tok/s, ITL, thinking tokens
      stats.py          ← Statistical analysis (p50, p95, p99, distributions)

    report/
      markdown.py       ← Markdown report: verdict, insights, grades, ASCII charts
```

---


## Contributing

We welcome contributions! Here's how to get started:

```bash
git clone https://github.com/swarmone/agentic-swarm-bench.git
cd agentic-swarm-bench

# With uv (recommended)
uv sync --all-extras
uv run pytest tests/ -v

# Or with pip
pip install -e ".[dev,proxy]"
make test
```

### Development

```bash
make lint      # Check code style
make format    # Auto-format
make test      # Run tests
```

### Adding tasks

Tasks are defined in `agentic_swarm_bench/tasks/tasks.json`. Each task has:

- `id`: P1 through P110
- `tier`: trivial, easy, medium, hard, expert
- `prompt`: the agentic swarm task
- `tags`: categorization (language, domain)
- `max_output_tokens`: token limit for the response

---

## License

Apache 2.0 - see [LICENSE](LICENSE).

---

<p align="center">
  <a href="https://swarmone.ai"><img src="./assets/swarm_one_logo.png" alt="SwarmOne" width="160" /></a>
</p>
<p align="center">
  <strong>Built by <a href="https://swarmone.ai">SwarmOne</a></strong>
</p>
