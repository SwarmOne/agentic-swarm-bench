# Contributing to AgenticSwarmBench

Thank you for your interest in contributing! This document explains how to get started.

## Development Setup

```bash
git clone https://github.com/swarmone/agentic-swarm-bench.git
cd agentic-swarm-bench
pip install -e ".[dev,proxy]"
```

## Running Tests

```bash
make test          # Run all tests
make lint          # Check for lint issues
make format        # Auto-format code
```

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length limit: 100 characters
- Use type hints on function signatures
- Write early returns instead of nested conditionals
- Optimize for readability - avoid cleverness

## Adding Tasks

Tasks live in `agentic_swarm_bench/tasks/tasks.json`. Each task needs:

| Field               | Description                            |
| ------------------- | -------------------------------------- |
| `id`                | P1 through P110                        |
| `tier`              | trivial, easy, medium, hard, or expert |
| `prompt`            | The swarm task description             |
| `tags`              | List of category tags                  |
| `max_output_tokens` | Max tokens for the response            |

Tier assignments follow these ranges:

- **Trivial** (P1-P10): Single function, < 20 lines
- **Easy** (P11-P25): Single file, 20-100 lines
- **Medium** (P26-P50): Multi-function, 100-300 lines
- **Hard** (P51-P75): Complex programs, 300-800 lines
- **Expert** (P76-P100): Multi-file projects, 800+ lines

## Pull Request Process

1. Fork and create a feature branch
2. Make your changes with tests
3. Run `make lint` and `make test`
4. Submit a PR with a clear description

## Reporting Issues

Please include:

- AgenticSwarmBench version (`asb --version`)
- Python version
- The command you ran
- Full error output
- Your endpoint type (vLLM, TGI, SGLang, etc.)
