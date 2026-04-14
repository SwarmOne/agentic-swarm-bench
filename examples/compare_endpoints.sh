#!/usr/bin/env bash
# Compare two endpoints side by side.
#
# Usage:
#   ./examples/compare_endpoints.sh \
#     http://endpoint-a:8000 model-a \
#     http://endpoint-b:8000 model-b

set -euo pipefail

ENDPOINT_A="${1:?Usage: $0 <endpointA> <modelA> <endpointB> <modelB>}"
MODEL_A="${2:?}"
ENDPOINT_B="${3:?}"
MODEL_B="${4:?}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/compare_${TIMESTAMP}"
mkdir -p "$OUTPUT"

echo "=== Benchmarking A: $MODEL_A ==="
agentic-swarm-bench speed \
  --endpoint "$ENDPOINT_A" \
  --model "$MODEL_A" \
  --suite standard \
  --output "$OUTPUT/a.md"

echo ""
echo "=== Benchmarking B: $MODEL_B ==="
agentic-swarm-bench speed \
  --endpoint "$ENDPOINT_B" \
  --model "$MODEL_B" \
  --suite standard \
  --output "$OUTPUT/b.md"

echo ""
echo "=== Generating comparison ==="
agentic-swarm-bench compare \
  --baseline "$OUTPUT/a.json" \
  --candidate "$OUTPUT/b.json" \
  --output "$OUTPUT/comparison.md"

echo ""
echo "Done! Comparison: $OUTPUT/comparison.md"
