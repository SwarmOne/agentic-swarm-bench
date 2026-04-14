<p align="center">
  <img src="https://raw.githubusercontent.com/swarmone/agentic-swarm-bench/main/assets/logo.png" alt="AgenticSwarmBench" width="640" />
</p>

<p align="center">
  <strong>The open-source benchmark for LLM inference under agentic swarm workloads</strong><br>
  Created by <a href="https://swarmone.ai"><img src="https://raw.githubusercontent.com/swarmone/agentic-swarm-bench/main/assets/swarmone-logo.svg" alt="SwarmOne" height="20" style="vertical-align: middle;" /></a> — the AI-native cloud for agentic workloads
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
  <a href="#benchmark-modes">Modes</a> &bull;
  <a href="#workload-recording--replay">Record & Replay</a> &bull;
  <a href="#the-110-tasks">Tasks</a> &bull;
  <a href="#context-control">Context Control</a> &bull;
  <a href="#reports">Reports</a> &bull;
  <a href="#docker">Docker</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarmone/agentic-swarm-bench/main/assets/demo.gif" alt="AgenticSwarmBench demo" width="720" />
</p>

---

## Why Agentic Swarm?

When Claude Code opens a file, reads 2,000 lines, edits three functions, runs tests, and reads the error output - that's **5+ LLM round-trips with 40-100K token contexts** growing each turn. Every turn adds tool results, file contents, and error traces to the conversation.

**No existing benchmark simulates this.**

- **SWE-bench** measures model quality on GitHub issues. It doesn't measure inference speed.
- **LMSys / Chatbot Arena** measures chatbot throughput at ~2K context. Agentic swarm contexts are 20-80x larger.
- **Generic LLM benchmarks** send uniform requests. Agentic swarm workloads have system prompts with tool schemas, multi-turn history, code files, and growing context windows.

**AgenticSwarmBench fills that gap** - it benchmarks your LLM serving stack under the exact access patterns that Claude Code, Cursor, Windsurf, and Copilot generate.

| What makes it different        |                                                                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic swarm context**      | Pads requests with real-looking agentic sessions - system prompts with tool definitions, prior conversation turns, code files, tool call results, error traces |
| **Growing context simulation** | Profiles simulate how context grows during a real coding session: fresh (6K) → short (20K) → medium (40K) → long (70K) → full (100K) → xl (200K) → xxl (400K) |
| **Prefix cache defeat**        | Unique per-request salt ensures you measure true cold-start inference, not cache hits                                                                         |
| **Cache impact measurement**   | `--cache-mode both` runs cold + warm to show exact prefix cache speedup (10x cost difference on Anthropic)                                                    |
| **Reasoning token detection**  | Automatically detects thinking/reasoning tokens (DeepSeek R1, o3, Claude Extended Thinking) and reports thinking overhead vs visible output latency           |
| **110 agentic swarm tasks**    | 5 difficulty tiers, 5 languages (Python, TypeScript, Rust, Go, SQL) - from single-function fixes to full-stack refactors                                      |
| **Record & replay**            | Capture real coding sessions as replayable workloads, then benchmark them against any endpoint                                                                |
| **Five CLI modes**             | Speed, eval, agent, record, and replay - plus reporting and comparison                                                                                        |
| **Docker one-liner**           | Point at any vLLM / SGLang / TGI / OpenAI-compatible endpoint and go                                                                                          |

---

## Quick Start

### Install

```bash
pip install agentic-swarm-bench
```

Or with proxy support (for agentic mode):

```bash
pip install "agentic-swarm-bench[proxy]"
```

### Run against your own endpoint

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

## Benchmark Modes

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              AgenticSwarmBench                                       │
├──────────────┬──────────────────┬──────────────────┬─────────────────────────────────┤
│  speed       │  eval            │  record / replay │  agent                          │
│              │                  │                  │                                 │
│  Direct to   │  Send swarm      │  Capture real    │  Recording proxy between        │
│  endpoint    │  tasks, validate │  agentic sessions│  agent and endpoint             │
│              │  generated code  │  as JSONL, then  │                                 │
│  Measures:   │  Measures:       │  replay anywhere │  Measures:                      │
│  TTFT        │  Syntax pass     │                  │  Real agentic session metrics   │
│  tok/s       │  Execution pass  │  Measures:       │  Multi-turn latency growth      │
│  ITL         │  Functional      │  Same as speed,  │  Tool call overhead             │
│  Prefill     │  correctness     │  from real data  │  Context window scaling         │
└──────────────┴──────────────────┴──────────────────┴─────────────────────────────────┘
```

### `asb speed` - Inference Speed Under Agentic Load

Sends streaming requests with realistic agentic swarm context (system prompts, tool schemas, file contents, conversation history) directly to any OpenAI-compatible endpoint.

```bash
# Default: realistic context sweep simulating a coding session growing over time
asb speed -e http://localhost:8000 -m my-model

# Specific concurrency (32 concurrent agents) at long context
asb speed -e http://localhost:8000 -m my-model -u 32 -p long

# Fixed token count - stress test at exactly 50K tokens
asb speed -e http://localhost:8000 -m my-model -c 50000 -u 16

# Cap max users - run a full suite but limit concurrency to 16
asb speed -e http://localhost:8000 -m my-model --suite full --max-users 16

# Measure prefix cache impact - runs cold then warm
asb speed -e http://localhost:8000 -m my-model --cache-mode both
```

```bash
# JSON-only output (for CI/CD pipelines)
asb speed -e http://localhost:8000 -m my-model --format json -o results.json

# Randomize context per request (tests diverse prefill patterns)
asb speed -e http://localhost:8000 -m my-model --random-context
```

**Metrics:** TTFT, decode tok/s per user, prefill tok/s, ITL (p50/p95/p99), aggregate throughput, reasoning token overhead. When the endpoint returns `prompt_tokens` in the response, actual token counts are shown alongside estimates.

### `asb eval` - Code Correctness

Sends agentic swarm tasks and validates the generated code at three levels:

```bash
# Syntax validation (does it parse?)
asb eval -e http://localhost:8000 -m my-model -t p1-p25 -v syntax

# Execution validation (does it run?)
asb eval -e http://localhost:8000 -m my-model -t p1-p25 -v execution

# Functional validation (does it produce correct output?)
asb eval -e http://localhost:8000 -m my-model -t p1-p25 -v functional
```

### `asb agent` - Full Agentic Session Benchmark

Runs a recording proxy between a real agent (Claude Code) and your endpoint, measuring actual multi-turn agentic sessions:

```bash
asb agent \
  -e http://localhost:8000 \
  -m my-model \
  -t p1-p10
```

The proxy translates Anthropic Messages API → OpenAI Chat Completions API and records per-request timing, context growth, and tool call patterns.

### `asb list-tasks` - Browse Available Tasks

```bash
asb list-tasks                        # Show all 110 tasks
asb list-tasks -t trivial             # Filter by tier
asb list-tasks --tags typescript,rust  # Filter by language
asb list-tasks --format json          # JSON output
```

---

## Workload Recording & Replay

Synthetic benchmarks are useful, but nothing beats measuring with **your actual coding sessions**. Record a real session, then replay it against any endpoint.

### `asb record` - Capture a Real Session

Starts a recording proxy between your agent and your LLM endpoint. Every request/response pair is saved as a JSONL line:

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

Both modes save the workload in OpenAI format for replay. Stop with `Ctrl+C` when done.

### `asb replay` - Replay Against Any Endpoint

Take a recorded workload and replay it against a different endpoint, hardware, or configuration:

```bash
# Replay a session against a new endpoint
asb replay \
  -e http://new-server:8000 \
  -m my-model \
  -w my-session.jsonl

# Generate a full report
asb replay \
  -e http://new-server:8000 \
  -m my-model \
  -w my-session.jsonl \
  -o report.md

# Preview without sending requests
asb replay -e URL -m MODEL -w session.jsonl --dry-run

# Replay just the beginning of a session (up to 1M cumulative prompt tokens)
asb replay -e URL -m MODEL -w session.jsonl --slice-tokens 1000000
```

**Slicing workloads:** Real sessions grow from small contexts to large ones. `--slice-tokens N` replays requests from the start until cumulative prompt tokens reach N - preserving the natural context growth while capping how much you send through the endpoint. Useful for targeting specific model context limits or keeping replay costs down.

Requests are grouped by context size and produce the same metrics as `asb speed` - TTFT, tok/s, ITL, and aggregate throughput.

### `asb list-workloads` - Browse Built-in Workloads

```bash
asb list-workloads
asb list-workloads --format json
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

## Prefix Cache Defeat

Every request includes a unique salt:

```
[session_id=abc123... ts=1234567890 rand=847291...]
```

This ensures prefix caching cannot mask cold-start prefill costs. Every measurement reflects true inference performance.

```bash
# Default: cache defeat enabled (cold-start measurement)
asb speed -e URL -m MODEL --defeat-cache

# Measure production-like performance with caching
asb speed -e URL -m MODEL --allow-cache

# Measure BOTH - shows exact cache speedup
asb speed -e URL -m MODEL --cache-mode both
```

`--cache-mode both` runs each scenario twice (first cold, then warm) and reports the delta. Anthropic charges 10x less for cached tokens ($0.30 vs $3.00/M), so knowing your cache hit rate matters.

## Reasoning Token Detection

Models like DeepSeek R1, o3, and Claude Extended Thinking produce **thinking tokens** before visible output. AgenticSwarmBench automatically detects `reasoning_content` in the streaming response and reports:

| Metric                | Description                                        |
| --------------------- | -------------------------------------------------- |
| **TTFT (thinking)**   | Time to first reasoning token                      |
| **TTFT (visible)**    | Time to first visible output token                 |
| **Thinking overhead** | Extra latency from reasoning before visible output |
| **Thinking tokens**   | Count of reasoning tokens generated                |

This is critical for agentic swarm workloads - a reasoning model that takes 5 seconds to "think" before emitting code changes the UX of the entire editing session.

---

## What Good Looks Like

Reference ranges from common setups (your numbers will vary by hardware, model size, and serving stack):

| Setup                                   | Context | Users |   TTFT | Tok/s/user | Notes                               |
| --------------------------------------- | ------- | ----: | -----: | ---------: | ----------------------------------- |
| vLLM on 1x A100 (80GB), 7B model        | 6K      |     1 | ~100ms |    ~80-120 | Baseline: fast model, short context |
| vLLM on 1x A100 (80GB), 7B model        | 40K     |     8 |  ~2-4s |     ~20-40 | Typical agentic workload            |
| vLLM on 1x A100 (80GB), 7B model        | 100K    |    32 | ~8-15s |      ~5-15 | Stress test                         |
| SGLang on 1x H100, 70B model            | 6K      |     1 | ~200ms |     ~40-60 | Larger model, faster GPU            |
| SGLang on 1x H100, 70B model            | 40K     |     8 |  ~3-6s |     ~10-25 | Agentic sweet spot                  |
| API provider (e.g. Together, Fireworks) | 40K     |     8 |  ~2-8s |     ~15-40 | Varies by provider/load             |

**Rules of thumb for agentic swarm workloads:**

- **TTFT < 3s** at 40K context → responsive editing experience
- **Tok/s > 30/user** → code appears to stream smoothly
- **TTFT < 10s** at 100K context → acceptable for deep sessions
- **Agg tok/s scales sub-linearly** with users - expect ~60-70% efficiency at 8x concurrency

> **A note on ITL:** Inter-Token Latency measures the gap between SSE `data:` lines as received by the client. Very low values (< 1ms) typically reflect HTTP/TCP buffering, not actual token generation speed. Use tok/s as the primary throughput metric; ITL is best for relative comparisons across scenarios.

---

## Reports

Reports are designed to answer one question fast: **is this endpoint good enough for agentic swarm workloads?**

Every report includes:

1. **Verdict** - 🟢 GOOD / 🟡 MARGINAL / 🔴 POOR for agentic swarm workloads, with the key numbers
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
  Verdict: GOOD for agentic swarm workloads at medium context
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
defeat_cache: true
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
| `ASB_DEFEAT_CACHE`         | Defeat prefix caching (`true`/`false`)              |

---

## Architecture

```
agentic-swarm-bench/
  agentic_swarm_bench/
    cli.py              ← Click CLI (asb speed | eval | agent | record | replay | ...)
    config.py           ← Config: CLI > env > YAML > defaults

    tasks/
      tasks.json        ← 110 agentic swarm tasks, P1-P110
      registry.py       ← Load/filter tasks by tier, range, tags, language
      context/
        codebase_context.py ← Agentic session context: tool schemas, file contents, cache defeat salt

    runner/
      direct.py         ← Speed mode: direct endpoint benchmark with agentic context
      eval_runner.py    ← Eval mode: code correctness validation
      claude_code.py    ← Agent mode: Claude Code orchestration through recording proxy

    workloads/
      recorder.py       ← Recording proxy: captures real sessions as JSONL workloads
      player.py         ← Replay engine: replays workloads against any endpoint
      registry.py       ← Load/list/resolve workloads (file path or built-in name)
      data/             ← Built-in workload files

    proxy/
      server.py         ← Agent-mode proxy (FastAPI) - Anthropic ↔ OpenAI translation
      padding.py        ← Context padding for proxy mode
      translators.py    ← API format translation

    metrics/
      collector.py      ← Per-request metrics: TTFT, tok/s, ITL, thinking tokens
      stats.py          ← Statistical analysis (p50, p95, p99, distributions)

    report/
      markdown.py       ← Markdown report: verdict, insights, grades, ASCII charts

  skill/
    SKILL.md            ← Claude Code skill: auto-optimize LLM deployments using asb
```

---

## Claude Code Optimization Skill

The repo includes a Claude Code skill (`skill/SKILL.md`) that turns Claude Code into an automated deployment optimizer. Point it at your serving stack and it will:

1. Run `asb speed` to establish a baseline
2. Read the verdict and key findings
3. Identify the bottleneck (prefill-bound, decode-bound, scheduling, or context scaling)
4. Tweak one deployment knob (tensor parallelism, batch size, chunked prefill, etc.)
5. Re-run and compare - repeat until targets are met or 5 iterations show no improvement

```bash
# Add the skill to Claude Code, then ask:
# "Optimize my vLLM deployment at http://localhost:8000 for agentic workload"
```

See `skill/SKILL.md` for the full skill definition and available knobs.

---

## Contributing

We welcome contributions! Here's how to get started:

```bash
git clone https://github.com/swarmone/agentic-swarm-bench.git
cd agentic-swarm-bench
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
