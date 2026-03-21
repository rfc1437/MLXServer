# MLX Server

Native macOS app for running local LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Built with SwiftUI, it provides both a **chat UI** and an embedded **OpenAI-compatible API server**. Supports both vision-capable and text-only MLX models, plus tool use and thinking mode where the selected model supports them.

## Supported Models

| Alias | Model | Context | Loader | Capabilities |
|-------|-------|---------|--------|-------------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | 128k | `VLMModelFactory` | Vision, tool use (`tool_code` blocks) |
| `qwen` | `mlx-community/Qwen3.5-4B-MLX-4bit` | 256k | `VLMModelFactory` | Vision, thinking mode, tool use (`<tool_call>` tags) |
| `qwen3.5-0.8b` | `mlx-community/Qwen3.5-0.8B-4bit` | 256k | `VLMModelFactory` | Vision, thinking mode, tool use (`<tool_call>` tags) |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-4bit` | 256k | `VLMModelFactory` | Vision, thinking mode, tool use (`<tool_call>` tags) |
| `stheno` | `synk/L3-8B-Stheno-v3.2-MLX` | 8k | `LLMModelFactory` | Text-only, llama-based |
| `violet-lotus` | `hobaratio/MN-Violet-Lotus-12B-mlx-4Bit` | 32k | `LLMModelFactory` | Text-only, Mistral-based |

Any model in MLX format on HuggingFace can be added — there is no restriction on uploader or architecture.

Developer note: the test suite uses `qwen3.5-0.8b` as the main live-model target because it is substantially faster and lighter than the larger Qwen variants, but some tests still run on Gemma 3 because they validate Gemma-specific prompt shaping, cache-reuse behavior, and tool-call behavior that did not match Qwen3.5 0.8B closely enough.

## Quick Start

Requires macOS 15+, Xcode 16.4+, and `xcodegen` (`brew install xcodegen`).

```bash
./build.sh            # Debug build
open "build/Debug/MLX Server.app"
```

Run tests with the repo entrypoint:

```bash
./test.sh
```

For focused test runs, `test.sh` also accepts `ONLY_TESTING` and forwards it to `xcodebuild -only-testing`:

```bash
ONLY_TESTING='MLXServerTests/ModelBackedInferenceValidationTests/testLarge4KImageUsesGemmaResizeConfigAndPreparesSuccessfully' ./test.sh
```

This is intended for targeted validation while keeping the normal default as the full suite.

## App Features

- **Chat interface** with markdown rendering and model-aware image attachments (file picker, drag & drop, clipboard paste, Finder copy-paste on vision-capable models)
- **Scene-based chat starts** — New Chat opens a scene picker with Neutral plus saved scenes, each with an optional model override, a scene prompt layered onto the base system prompt, an auto-sent starter prompt, and optional generation-setting overrides for chat-specific behavior
- **Model picker** in toolbar with curated defaults plus any locally discovered MLX models on disk
- **Models window** in the menu for downloading a model by HuggingFace ID, inspecting on-disk model sizes, and deleting local model folders
- **Download progress modal** — shows file progress, percentage, and speed when downloading a new model
- **Thinking mode** — models like Qwen3.5 can reason internally before responding; thinking content appears in a collapsible box. Toggle on/off in Settings.
- **Streaming responses** with live token display
- **Native chat documents** — save chats as `.mlxchat` package documents, reopen them from File > Open Chat or by double-clicking them in Finder, and continue the conversation with restored model context, thinking blocks, and images
- **Export chat** — File > Export Chat (Cmd+Shift+E) saves conversations as Markdown or RTF (Pages-compatible)
- **Status bar** showing model name, context window, tokens/sec, token counts, GPU memory, API server status
- **Keyboard shortcuts**: `Cmd+N` (new chat), `Cmd+O` (open chat document), `Cmd+S` (save chat document), `Cmd+Shift+S` (save chat document as), `Cmd+Shift+E` (export), `Cmd+Return` (send), `Escape` (stop), `Cmd+1/2/3/4/5` (switch models)
- **Scene management** — create and edit reusable roleplay/task presets from the New Chat flow or Settings
- **Settings** (`Cmd+,`): default model, per-model generation defaults (temperature, top-p/top-k, min-p, repetition/presence/frequency penalties, max tokens, thinking mode), base system prompt, scene management, API port, API auto-start, idle unload timeout
- **Idle auto-unload** — model is unloaded after configurable idle time (resets on both user input and model output), reloaded on next request

## API Server

The embedded API server (toggle in toolbar) runs on port 1234 by default. Standard OpenAI-compatible endpoints:

- `GET /v1/models` — lists available models with `context_window` sizes
- `POST /v1/chat/completions` — chat completions (streaming and non-streaming)
- `GET /health` — health check

Capability checks are enforced server-side. If a request sends images to a text-only model or tools to a model without tool support, the server returns a `400 invalid_request_error`.

When a chat-completions request omits generation parameters, the API server falls back to the saved per-model defaults from Settings. Request-supplied values still take precedence on a per-call basis.

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

Text-only models such as `stheno` reject image inputs.

### Tool Use

Pass tools in the `tools` field (OpenAI format). The server handles model-specific formatting (Gemma `tool_code` blocks, Qwen `<tool_call>` XML tags) and parses tool calls from output automatically. When tools are present during streaming, output is buffered to strip tool-call markup before sending to the client.

`stheno` is currently documented and configured as a plain text model, so tool requests to it are rejected.

## Project Structure

```
MLXServer/
├── MLXServerApp.swift              — App entry point, GPU cache config
├── ContentView.swift               — Main layout, toolbar, keyboard shortcuts
├── Models/
│   ├── ModelConfig.swift           — Model definitions, alias/repoId resolution
│   └── ChatMessage.swift           — Chat message data model, thinking tag parser
│   └── ChatScene.swift             — Persisted chat scene presets (prompt + model + starter)
├── ViewModels/
│   ├── ModelManager.swift          — Model loading/switching, download tracking, idle unload
│   └── ChatViewModel.swift         — Chat state, ChatSession, API server lifecycle
│   └── SceneStore.swift            — Scene persistence and editing operations
├── Views/
│   ├── SceneSelectionView.swift    — New chat scene picker popover
│   ├── SceneManagementView.swift   — Scene editor and list management
│   ├── ModelPickerView.swift       — Toolbar model selector with re-download
│   ├── ChatMessagesView.swift      — Scrollable message list with markdown + thinking blocks
│   ├── ChatInputView.swift         — Text input + image attach (paste, drag, picker)
│   ├── DownloadModalView.swift     — Model download progress overlay
│   ├── StatusBarView.swift         — Model info, tok/s, GPU memory, API status
│   ├── MonitorView.swift           — Inference statistics monitor
│   └── SettingsView.swift          — System prompt, thinking mode, API, idle settings
├── Commands/
│   └── SaveChatCommands.swift      — File menu new/open/save/revert/export commands
├── Documents/
│   ├── ChatDocumentController.swift — Queues Finder/app open-document requests into SwiftUI
│   ├── ChatDocumentManifest.swift   — Versioned `.mlxchat` manifest schema
│   ├── ChatDocumentMigration.swift  — Manifest schema migration entry point
│   └── ChatDocumentPackage.swift    — Package document read/write for `.mlxchat`
├── Server/
│   ├── APIServer.swift             — NWListener HTTP server, SSE streaming, KV cache reuse
│   ├── APIModels.swift             — OpenAI-compatible Codable structs
│   ├── ToolCallParser.swift        — Parses tool calls from model output
│   └── ToolPromptBuilder.swift     — Model-specific tool prompt formatting
└── Utilities/
    ├── LocalModelResolver.swift    — Offline-first HuggingFace cache resolution (sandbox + system)
    ├── ChatExporter.swift          — Export conversations to Markdown or RTF
    ├── FocusedValues.swift         — FocusedValue keys for menu bar integration
    └── Preferences.swift           — UserDefaults wrapper, including scene persistence

project.yml     — xcodegen project spec (dependencies, settings, deployment target)
build.sh        — One-command build script (xcodegen + xcodebuild)
```

## Key Design Decisions

- Uses `mlx-swift-lm` for inference — `VLMModelFactory` for vision models and `LLMModelFactory` for text-only models
- **Offline-first**: `LocalModelResolver` checks both the sandboxed app container and `~/.cache/huggingface/hub/` for locally-cached models before downloading
- **No duplicate storage**: custom `HubApi` with blob cache disabled — models are stored once in the snapshot cache
- **KV cache reuse** across API requests — reuses `ChatSession` when conversation history prefix matches
- **Thinking mode**: `enable_thinking` passed via Jinja template context; `<think>` tags parsed in real-time during streaming
- HTTP server built on `Network.framework` (`NWListener`) — no third-party server dependencies
- Model-specific prompt formatting: Gemma uses `tool_code` blocks, Qwen uses `<tool_call>` XML tags
- GPU cache limit set to 20 MB; cache cleared on model unload
