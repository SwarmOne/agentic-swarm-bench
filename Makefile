.PHONY: install dev lint test clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev,proxy]"

lint:
	ruff check agentic_swarm_bench/ tests/
	ruff format --check agentic_swarm_bench/ tests/

format:
	ruff check --fix agentic_swarm_bench/ tests/
	ruff format agentic_swarm_bench/ tests/

test:
	pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t swarmone/agentic-swarm-bench .
