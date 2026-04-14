#!/usr/bin/env bash
# Quick start: run a fast benchmark against a local endpoint.
# This runs the "quick" suite (1 and 8 users, medium context).
#
# Usage:
#   ./examples/quick_start.sh http://localhost:8000 my-model-name

set -euo pipefail

ENDPOINT="${1:?Usage: $0 <endpoint> <model>}"
MODEL="${2:?Usage: $0 <endpoint> <model>}"

echo "=== AgenticSwarmBench - Quick Start ==="
echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo ""

agentic-swarm-bench speed \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --suite quick \
  --output results/quick_report.md

echo ""
echo "Done! Report: results/quick_report.md"
