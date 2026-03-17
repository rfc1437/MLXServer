# MLX Server

OpenAI-compatible API server for Gemma 3 4B (vision + tool use) on Apple Silicon via MLX.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the server (downloads model on first run)
./run.sh

# Or directly:
python -m mlx_server.main --model mlx-community/gemma-3-4b-it-4bit --port 1234
```

## Project Structure

- `mlx_server/main.py` — FastAPI server, endpoints, CLI entrypoint
- `mlx_server/engine.py` — Model loading, prompt building, generation (mlx_vlm)
- `mlx_server/models.py` — Pydantic models for OpenAI API request/response types

## Key Design Decisions

- Uses `mlx_vlm` (not `mlx_lm`) as the inference backend — this supports both text and vision in a single model load
- Gemma 3 has no system role — system messages are converted to user/assistant pairs
- Tool use is prompt-engineered: tools are injected into the system prompt with `<tool_call>` XML tags, and parsed from model output
- Thread lock on generation (single-request-at-a-time) — MLX models aren't safe for concurrent generation
- 128k context window supported via the model's native capabilities

## Dependencies

Managed via `uv` and `pyproject.toml`. Virtual environment in `.venv/`.

```bash
uv pip install -e "."
```
