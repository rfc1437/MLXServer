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

## Critical Performance Rule

**Inference speed is the #1 priority.** The token generation loop must never be blocked or slowed by anything else — no MainActor hops, no SwiftUI observation, no synchronous I/O. Everything that isn't inference (stats collection, UI updates, logging) must run on separate threads via loose coupling:

- **`LiveCounters`** (thread-safe singleton with `OSAllocatedUnfairLock`) is the bridge: generation code writes to it directly from any thread with zero actor overhead.
- **`InferenceStats`** (UI-side, `@Observable @MainActor`) polls `LiveCounters` at 1Hz via a timer — never the other way around.
- SSE streaming (`sendSSEEvent`/`sendData`) runs nonisolated off MainActor so token sends don't compete with SwiftUI rendering.
- Never gate token output on UI state, analytics, or any `@MainActor`-isolated code.

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
