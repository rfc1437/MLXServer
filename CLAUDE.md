# MLX Server

Native macOS SwiftUI app for local LLMs on Apple Silicon via MLX. Provides a chat UI and an embedded OpenAI-compatible API server. Supports vision and tool use.

## Quick Start

```bash
# Build (requires xcodegen: brew install xcodegen)
./build.sh

# Run
open "build/Debug/MLX Server.app"
```

## Project Structure

- `MLXServer/MLXServerApp.swift` — App entry point, GPU cache config
- `MLXServer/ContentView.swift` — Main layout, toolbar, keyboard shortcuts
- `MLXServer/Models/ModelConfig.swift` — Model definitions (alias, repoId, contextLength), resolution
- `MLXServer/Models/ChatMessage.swift` — Chat message data model
- `MLXServer/ViewModels/ModelManager.swift` — Model loading/switching via VLMModelFactory, offline-first resolution
- `MLXServer/ViewModels/ChatViewModel.swift` — Chat state, ChatSession management, API server lifecycle
- `MLXServer/Server/APIServer.swift` — NWListener HTTP server, SSE streaming, KV cache reuse, vision, tool call handling
- `MLXServer/Server/APIModels.swift` — OpenAI-compatible Codable structs
- `MLXServer/Server/ToolCallParser.swift` — Parses tool calls from model output (Gemma tool_code, Qwen XML tags)
- `MLXServer/Server/ToolPromptBuilder.swift` — Model-specific tool prompt formatting
- `MLXServer/Utilities/LocalModelResolver.swift` — Resolves HF repo IDs to ~/.cache/huggingface/hub/ snapshots
- `MLXServer/Utilities/Preferences.swift` — UserDefaults wrapper
- `project.yml` — xcodegen project spec
- `build.sh` — Build script (xcodegen + xcodebuild)

## Supported Models

| Alias | HuggingFace ID | Notes |
|-------|---------------|-------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | Vision + tool use via `tool_code` blocks (128k context) |
| `qwen` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | Vision + tool use via `<tool_call>` tags (256k context) |

## Key Design Decisions

- Uses `mlx-swift-lm` (`MLXVLM` / `VLMModelFactory`) as the inference backend — supports both text and vision in a single model load
- Model-specific prompt formatting: Gemma uses `tool_code` blocks; Qwen uses `<tool_call>` XML tags
- Offline-first: if the model is already cached locally (~/.cache/huggingface/hub/), `LocalModelResolver` resolves the local snapshot path directly — no network requests
- HTTP server built on `Network.framework` (`NWListener`) — no third-party server dependencies
- KV cache reuse across API requests — reuses `ChatSession` when conversation history prefix matches
- GPU cache limit set to 20 MB; cache cleared on model unload

## Dependencies

Managed via Swift Package Manager (declared in `project.yml` for xcodegen).

| Package | Products |
|---------|----------|
| `mlx-swift-lm` | `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| `swift-markdown-ui` | `MarkdownUI` |
