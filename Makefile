.PHONY: install dev lint test clean docker

install:
	uv sync

dev:
	uv sync --all-extras

lint:
	uv run ruff check agentic_swarm_bench/ tests/
	uv run ruff format --check agentic_swarm_bench/ tests/

format:
	uv run ruff check --fix agentic_swarm_bench/ tests/
	uv run ruff format agentic_swarm_bench/ tests/

test:
	uv run pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t swarmone/agentic-swarm-bench .
