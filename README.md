# MLX Server

Native macOS app for running local LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Built with SwiftUI, it provides both a **chat UI** and an embedded **OpenAI-compatible API server**. Supports vision and tool use with automatic model swapping.

## Supported Models

| Alias | Model | Context | Capabilities |
|-------|-------|---------|-------------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | 128k | Vision, tool use (`tool_code` blocks) |
| `qwen` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 256k | Vision, tool use (`<tool_call>` tags) |

## Quick Start

Requires macOS 15+, Xcode 16.4+, and `xcodegen` (`brew install xcodegen`).

```bash
./build.sh            # Debug build
open "build/Debug/MLX Server.app"
```

## App Features

- **Chat interface** with markdown rendering, image attachments (file picker, drag & drop, clipboard paste)
- **Model picker** in toolbar with local/download status indicators
- **Streaming responses** with live token display
- **Status bar** showing model name, context window, tokens/sec, token counts, GPU memory, API server status
- **Keyboard shortcuts**: `Cmd+N` (new chat), `Cmd+Return` (send), `Escape` (stop), `Cmd+1/2/3` (switch models)
- **Settings** (`Cmd+,`): system prompt, API port, API auto-start

## API Server

The embedded API server (toggle in toolbar) runs on port 1234 by default. Standard OpenAI-compatible endpoints:

- `GET /v1/models` — lists available models with `context_window` sizes
- `POST /v1/chat/completions` — chat completions (streaming and non-streaming)
- `GET /health` — health check

### Model Swapping

Send any model ID or alias in the `model` field. If it differs from the currently loaded model, the server swaps automatically:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Vision

Pass images as base64 data URIs in the `image_url` content part:

```json
{
  "model": "gemma",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

### Tool Use

Pass tools in the `tools` field (OpenAI format). The server handles model-specific formatting (Gemma `tool_code` blocks, Qwen `<tool_call>` XML tags) and parses tool calls from output automatically. When tools are present during streaming, output is buffered to strip tool-call markup before sending to the client.

## Project Structure

```
MLXServer/
├── MLXServerApp.swift              — App entry point, GPU cache config
├── ContentView.swift               — Main layout, toolbar, keyboard shortcuts
├── Models/
│   ├── ModelConfig.swift           — Model definitions, alias/repoId resolution
│   └── ChatMessage.swift           — Chat message data model
├── ViewModels/
│   ├── ModelManager.swift          — Model loading/switching via VLMModelFactory
│   └── ChatViewModel.swift         — Chat state, ChatSession, API server lifecycle
├── Views/
│   ├── ModelPickerView.swift       — Toolbar model selector
│   ├── ChatMessagesView.swift      — Scrollable message list with markdown
│   ├── ChatInputView.swift         — Text input + image attach
│   ├── StatusBarView.swift         — Model info, tok/s, GPU memory, API status
│   └── SettingsView.swift          — System prompt + API settings
├── Server/
│   ├── APIServer.swift             — NWListener HTTP server, SSE streaming, KV cache reuse
│   ├── APIModels.swift             — OpenAI-compatible Codable structs
│   ├── ToolCallParser.swift        — Parses tool calls from model output
│   └── ToolPromptBuilder.swift     — Model-specific tool prompt formatting
└── Utilities/
    ├── LocalModelResolver.swift    — Offline-first HuggingFace cache resolution
    └── Preferences.swift           — UserDefaults wrapper

project.yml     — xcodegen project spec (dependencies, settings, deployment target)
build.sh        — One-command build script (xcodegen + xcodebuild)
```

## Key Design Decisions

- Uses `mlx-swift-lm` (`MLXVLM` / `VLMModelFactory`) for inference — supports both text and vision in a single model load
- **Offline-first**: `LocalModelResolver` checks `~/.cache/huggingface/hub/` for locally-cached snapshots before downloading
- **KV cache reuse** across API requests — reuses `ChatSession` when conversation history prefix matches
- HTTP server built on `Network.framework` (`NWListener`) — no third-party server dependencies
- Model-specific prompt formatting: Gemma uses `tool_code` blocks, Qwen uses `<tool_call>` XML tags
- GPU cache limit set to 20 MB; cache cleared on model unload

## Design Notes

- Uses `mlx_vlm` (not `mlx_lm`) as the backend — supports both text and vision in a single model load
- Offline-first: if the model is cached locally (`~/.cache/huggingface/hub/`), no network requests are made
- Thread lock on generation — MLX models aren't safe for concurrent generation
- KV prefix caching for multi-turn conversations
- Context window read from each model's config (Gemma 3 4B: 128k, Qwen3-VL 4B: 256k) with automatic summarization fallback
