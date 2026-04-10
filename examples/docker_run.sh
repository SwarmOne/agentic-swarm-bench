#!/usr/bin/env bash
# Run agentic-coding-bench in Docker against any endpoint.
#
# Usage:
#   ./examples/docker_run.sh http://host.docker.internal:8000 my-model

set -euo pipefail

ENDPOINT="${1:?Usage: $0 <endpoint> <model>}"
MODEL="${2:?Usage: $0 <endpoint> <model>}"

echo "=== AgenticCodingBench (Docker) ==="
echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo ""

docker run --rm \
  -v "$(pwd)/results:/results" \
  swarmone/agentic-coding-bench speed \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --suite quick \
    --output /results/docker_report.md

echo ""
echo "Done! Report: results/docker_report.md"
