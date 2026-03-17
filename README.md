# MLX Server

OpenAI-compatible API server for running local LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Supports vision and tool use with automatic model swapping — only one model is loaded in memory at a time, switched on demand based on the request's `model` field.

## Supported Models

| Alias | Model | Context | Capabilities |
|-------|-------|---------|-------------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | 128k | Vision, tool use (`tool_code` blocks) |
| `gemma3n` | `mlx-community/gemma-3n-E4B-it-4bit` | 32k | Vision/audio/video, tool use (`tool_code` blocks), ~1.5x faster |
| `qwen` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 256k | Vision, tool use (`<tool_call>` tags) |

## Quick Start

```bash
source .venv/bin/activate

# Start with Gemma 3 (default)
./run.sh

# Start with Qwen3
./run.sh qwen

# Or directly
python -m mlx_server.main --model mlx-community/gemma-3-4b-it-4bit --port 1234
```

The server starts at `http://127.0.0.1:1234`.

## API

Standard OpenAI-compatible endpoints:

- `GET /v1/models` — lists all available models with `context_window` sizes
- `POST /v1/chat/completions` — chat completions (streaming and non-streaming)
- `GET /health` — health check

### Model Swapping

Send any available model ID (or alias) in the `model` field. If it differs from the currently loaded model, the server unloads the old one and loads the new one automatically:

```bash
# Uses Gemma
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-3-4b-it-4bit", "messages": [{"role": "user", "content": "Hello"}]}'

# Swaps to Qwen
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-VL-4B-Instruct-4bit", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Vision

Pass images as base64 data URIs or URLs in the `image_url` content part:

```json
{
  "model": "mlx-community/gemma-3-4b-it-4bit",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

### Context Window Management

Each model's context window is read from its HuggingFace config (`max_position_embeddings`) and reported in `/v1/models` via the `context_window` field. Clients can use this to manage conversation length proactively.

If a request exceeds the context window, the server:

1. Automatically summarizes older messages (keeping system messages and the last 6 messages intact)
2. Retries with the compressed conversation
3. Returns an OpenAI-compatible `context_length_exceeded` error if it still doesn't fit

### Tool Use

Pass tools in the `tools` field (OpenAI format). The server handles model-specific formatting and parses tool calls from the output automatically.

## Installation

Requires Python 3.11+ and Apple Silicon.

```bash
uv pip install -e "."
```

## Project Structure

```
mlx_server/
  main.py    — FastAPI server, endpoints, CLI entrypoint
  engine.py  — Model loading, prompt building, generation (mlx_vlm)
  models.py  — Pydantic models for OpenAI API types
```

## Design Notes

- Uses `mlx_vlm` (not `mlx_lm`) as the backend — supports both text and vision in a single model load
- Offline-first: if the model is cached locally (`~/.cache/huggingface/hub/`), no network requests are made
- Thread lock on generation — MLX models aren't safe for concurrent generation
- KV prefix caching for multi-turn conversations
- Context window read from each model's config (Gemma 3 4B: 128k, Qwen3-VL 4B: 256k) with automatic summarization fallback
