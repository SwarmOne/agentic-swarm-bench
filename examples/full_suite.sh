#!/usr/bin/env bash
# Full benchmark suite: all context profiles and concurrency levels.
# This takes significantly longer but gives a complete picture.
#
# Usage:
#   ./examples/full_suite.sh http://localhost:8000 my-model

set -euo pipefail

ENDPOINT="${1:?Usage: $0 <endpoint> <model>}"
MODEL="${2:?Usage: $0 <endpoint> <model>}"
OUTPUT="results/${MODEL//\//_}_full_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT"

echo "=== AgenticSwarmBench - Full Suite ==="
echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo "Output:   $OUTPUT"
echo ""

agentic-swarm-bench speed \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --suite full \
  --tasks p1-p50 \
  --output "$OUTPUT/report.md"

echo ""
echo "Done! Results in $OUTPUT/"
