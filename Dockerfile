FROM python:3.12-slim

LABEL maintainer="SwarmOne <eng@swarmone.ai>"
LABEL org.opencontainers.image.source="https://github.com/swarmone/agentic-swarm-bench"
LABEL org.opencontainers.image.description="Benchmark LLM inference for agentic swarm workloads"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY agentic_swarm_bench/ agentic_swarm_bench/
COPY assets/ assets/

RUN pip install --no-cache-dir ".[proxy]"

RUN mkdir -p /results /traces

ENTRYPOINT ["agentic-swarm-bench"]
CMD ["--help"]
