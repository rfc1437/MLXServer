#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Default model – 4-bit quantized Gemma 3 4B IT (vision-capable)
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1234}"

echo "Starting MLX Server..."
echo "  Model: $MODEL"
echo "  Endpoint: http://$HOST:$PORT"
echo ""

exec python -m mlx_server.main \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
