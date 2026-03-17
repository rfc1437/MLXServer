#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# --- Model selection ---
# Usage:  ./run.sh [gemma|qwen]
# Or set MODEL env var directly for a custom model.

MODEL_CHOICE="${1:-gemma}"

if [[ -z "${MODEL:-}" ]]; then
    case "$MODEL_CHOICE" in
        gemma)
            MODEL="mlx-community/gemma-3-4b-it-4bit"
            ;;
        qwen)
            MODEL="mlx-community/Qwen3-VL-4B-Instruct-4bit"
            ;;
        *)
            echo "Unknown model choice: $MODEL_CHOICE"
            echo "Usage: $0 [gemma|qwen]"
            exit 1
            ;;
    esac
fi

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
    "${@:2}"
