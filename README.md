# MLX Server

Native macOS app for running local LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Built with SwiftUI, it provides both a **chat UI** and an embedded **OpenAI-compatible API server**. Supports vision, tool use, and thinking mode.

## Supported Models

| Alias | Model | Context | Capabilities |
|-------|-------|---------|-------------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | 128k | Vision, tool use (`tool_code` blocks) |
| `qwen` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 256k | Vision, tool use (`<tool_call>` tags) |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-4bit` | 256k | Thinking mode, tool use |

Any model in MLX format on HuggingFace can be added — there is no restriction on uploader or architecture.

## Quick Start

Requires macOS 15+, Xcode 16.4+, and `xcodegen` (`brew install xcodegen`).

```bash
./build.sh            # Debug build
open "build/Debug/MLX Server.app"
```

## App Features

- **Chat interface** with markdown rendering, image attachments (file picker, drag & drop, clipboard paste, Finder copy-paste)
- **Model picker** in toolbar with local/download status indicators and re-download button
- **Download progress modal** — shows file progress, percentage, and speed when downloading a new model
- **Thinking mode** — models like Qwen3.5 can reason internally before responding; thinking content appears in a collapsible box. Toggle on/off in Settings.
- **Streaming responses** with live token display
- **Export chat** — File > Export Chat (Cmd+Shift+S) saves conversations as Markdown or RTF (Pages-compatible)
- **Status bar** showing model name, context window, tokens/sec, token counts, GPU memory, API server status
- **Keyboard shortcuts**: `Cmd+N` (new chat), `Cmd+Return` (send), `Escape` (stop), `Cmd+1/2/3/4` (switch models), `Cmd+Shift+S` (export)
- **Settings** (`Cmd+,`): default model, thinking mode toggle, system prompt, API port, API auto-start, idle unload timeout
- **Idle auto-unload** — model is unloaded after configurable idle time (resets on both user input and model output), reloaded on next request

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
│   └── ChatMessage.swift           — Chat message data model, thinking tag parser
├── ViewModels/
│   ├── ModelManager.swift          — Model loading/switching, download tracking, idle unload
│   └── ChatViewModel.swift         — Chat state, ChatSession, API server lifecycle
├── Views/
│   ├── ModelPickerView.swift       — Toolbar model selector with re-download
│   ├── ChatMessagesView.swift      — Scrollable message list with markdown + thinking blocks
│   ├── ChatInputView.swift         — Text input + image attach (paste, drag, picker)
│   ├── DownloadModalView.swift     — Model download progress overlay
│   ├── StatusBarView.swift         — Model info, tok/s, GPU memory, API status
│   ├── MonitorView.swift           — Inference statistics monitor
│   └── SettingsView.swift          — System prompt, thinking mode, API, idle settings
├── Commands/
│   └── SaveChatCommands.swift      — File menu export command
├── Server/
│   ├── APIServer.swift             — NWListener HTTP server, SSE streaming, KV cache reuse
│   ├── APIModels.swift             — OpenAI-compatible Codable structs
│   ├── ToolCallParser.swift        — Parses tool calls from model output
│   └── ToolPromptBuilder.swift     — Model-specific tool prompt formatting
└── Utilities/
    ├── LocalModelResolver.swift    — Offline-first HuggingFace cache resolution (sandbox + system)
    ├── ChatExporter.swift          — Export conversations to Markdown or RTF
    ├── FocusedValues.swift         — FocusedValue keys for menu bar integration
    └── Preferences.swift           — UserDefaults wrapper

project.yml     — xcodegen project spec (dependencies, settings, deployment target)
build.sh        — One-command build script (xcodegen + xcodebuild)
```

## Key Design Decisions

- Uses `mlx-swift-lm` (`MLXVLM` / `VLMModelFactory`) for inference — loads any MLX-format model from HuggingFace
- **Offline-first**: `LocalModelResolver` checks both the sandboxed app container and `~/.cache/huggingface/hub/` for locally-cached models before downloading
- **No duplicate storage**: custom `HubApi` with blob cache disabled — models are stored once in the snapshot cache
- **KV cache reuse** across API requests — reuses `ChatSession` when conversation history prefix matches
- **Thinking mode**: `enable_thinking` passed via Jinja template context; `<think>` tags parsed in real-time during streaming
- HTTP server built on `Network.framework` (`NWListener`) — no third-party server dependencies
- Model-specific prompt formatting: Gemma uses `tool_code` blocks, Qwen uses `<tool_call>` XML tags
- GPU cache limit set to 20 MB; cache cleared on model unload
