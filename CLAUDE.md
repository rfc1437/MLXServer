# MLX Server

OpenAI-compatible API server for local LLMs on Apple Silicon via MLX. Supports Gemma 3 4B and Qwen3 VL 4B (vision + tool use).

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with Gemma 3 (default)
./run.sh

# Run with Qwen3
./run.sh qwen

# Or directly:
python -m mlx_server.main --model mlx-community/gemma-3-4b-it-4bit --port 1234
python -m mlx_server.main --model mlx-community/Qwen3-VL-4B-Instruct-4bit --port 1234
```

## Project Structure

- `mlx_server/main.py` — FastAPI server, endpoints, CLI entrypoint
- `mlx_server/engine.py` — Model loading, prompt building, generation (mlx_vlm)
- `mlx_server/models.py` — Pydantic models for OpenAI API request/response types

## Supported Models

| Alias | HuggingFace ID | Notes |
|-------|---------------|-------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | Vision + tool use via `tool_code` blocks (128k context) |
| `gemma3n` | `mlx-community/gemma-3n-E4B-it-4bit` | Vision/audio/video + tool use via `tool_code` blocks (32k context, ~1.5x faster) |
| `qwen` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | Vision + tool use via `<tool_call>` tags (256k context) |

## Key Design Decisions

- Uses `mlx_vlm` (not `mlx_lm`) as the inference backend — this supports both text and vision in a single model load
- Model-specific prompt formatting: Gemma converts system→user/assistant pairs and uses `tool_code` blocks; Qwen3 uses native system role and `<tool_call>` XML tags
- Offline-first: if the model is already cached locally (~/.cache/huggingface/hub/), the server resolves the local snapshot path directly — no network requests are made (HEAD checks, update checks, etc.)
- Thread lock on generation (single-request-at-a-time) — MLX models aren't safe for concurrent generation
- Context window size is read from each model's config at load time (Gemma 3 4B: 128k, Qwen3-VL 4B: 256k)

## Dependencies

Managed via `uv` and `pyproject.toml`. Virtual environment in `.venv/`.

```bash
uv pip install -e "."
```
