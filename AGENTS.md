# MLX Server

Native macOS SwiftUI app for local LLMs on Apple Silicon via MLX. Provides a chat UI and an embedded OpenAI-compatible API server. Supports vision, tool use, and thinking mode.

## Quick Start

**Always use `./build.sh` to build the project** — never call `xcodebuild` directly. The script runs xcodegen first (to pick up new/removed files) and uses the correct scheme, destination, and build directory.

**Always use `./test.sh` to run tests** — it regenerates the Xcode project first and runs the shared `MLXServer` test scheme so test runs are reproducible.

Tests are required for finished work when the change is reasonably testable.
Relevant tests must exist and must pass before work is considered complete.

Pre-existing errors don't exist: every error is your responsibility and you have to fix it before claiming you are done.

```bash
# Build (requires xcodegen: brew install xcodegen)
./build.sh

# Test
./test.sh

# Run
open "build/Debug/MLX Server.app"
```

## Project Structure

- `MLXServer/MLXServerApp.swift` — App entry point, GPU cache config, menu commands
- `MLXServer/ContentView.swift` — Main layout, toolbar, keyboard shortcuts, focused values
- `MLXServer/Models/ModelConfig.swift` — Model definitions (alias, repoId, contextLength), resolution
- `MLXServer/Models/ChatMessage.swift` — Chat message data model, `<think>` tag parsing
- `MLXServer/ViewModels/ModelManager.swift` — Model loading/switching via VLMModelFactory, download tracking, idle unload
- `MLXServer/ViewModels/ChatViewModel.swift` — Chat state, ChatSession management, API server lifecycle
- `MLXServer/Server/APIServer.swift` — NWListener HTTP server, SSE streaming, KV cache reuse, vision, tool call handling
- `MLXServer/Server/APIModels.swift` — OpenAI-compatible Codable structs
- `MLXServer/Server/ToolCallParser.swift` — Parses tool calls from model output (Gemma tool_code, Qwen XML tags)
- `MLXServer/Server/ToolPromptBuilder.swift` — Model-specific tool prompt formatting
- `MLXServer/Views/DownloadModalView.swift` — Modal overlay for model download progress
- `MLXServer/Views/ChatMessagesView.swift` — Message bubbles with markdown rendering and collapsible thinking blocks
- `MLXServer/Views/ChatInputView.swift` — Text input, image attach (file picker, drag & drop, Finder copy-paste)
- `MLXServer/Commands/SaveChatCommands.swift` — File > Export Chat menu command
- `MLXServer/Utilities/LocalModelResolver.swift` — Resolves HF repo IDs to local snapshots (sandbox + system cache + flat layouts)
- `MLXServer/Utilities/ChatExporter.swift` — Export conversations to Markdown or RTF (Pages-compatible)
- `MLXServer/Utilities/FocusedValues.swift` — FocusedValue keys for menu bar integration
- `MLXServer/Utilities/Preferences.swift` — UserDefaults wrapper (model, thinking mode, API, idle timeout)
- `project.yml` — xcodegen project spec
- `build.sh` — Build script (xcodegen + xcodebuild)

## Supported Models

| Alias | HuggingFace ID | Notes |
|-------|---------------|-------|
| `gemma` | `mlx-community/gemma-3-4b-it-4bit` | Vision + tool use via `tool_code` blocks (128k context) |
| `qwen` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Vision + thinking mode + tool use via `<tool_call>` tags (256k context) |
| `qwen3.5-0.8b` | `mlx-community/Qwen3.5-0.8B-4bit` | Vision + thinking mode + tool use via `<tool_call>` tags (256k context) |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-4bit` | Vision + thinking mode + tool use via `<tool_call>` tags (256k context) |
| `violet-lotus` | `hobaratio/MN-Violet-Lotus-12B-mlx-4Bit` | Text-only Mistral-based model (32k context) |

Any model in MLX format on HuggingFace can be added — no restriction on uploader or architecture.

## Critical Performance Rule

**Inference speed is the #1 priority.** The token generation loop must never be blocked or slowed by anything else — no MainActor hops, no SwiftUI observation, no synchronous I/O. Everything that isn't inference (stats collection, UI updates, logging) must run on separate threads via loose coupling:

- **`LiveCounters`** (thread-safe singleton with `OSAllocatedUnfairLock`) is the bridge: generation code writes to it directly from any thread with zero actor overhead.
- **`InferenceStats`** (UI-side, `@Observable @MainActor`) polls `LiveCounters` at 1Hz via a timer — never the other way around.
- SSE streaming (`sendSSEEvent`/`sendData`) runs nonisolated off MainActor so token sends don't compete with SwiftUI rendering.
- Never gate token output on UI state, analytics, or any `@MainActor`-isolated code.

## Key Design Decisions

- Uses `mlx-swift-lm` (`MLXVLM` / `VLMModelFactory`) as the inference backend — loads any MLX-format model from HuggingFace
- Model-specific prompt formatting: Gemma uses `tool_code` blocks; Qwen uses `<tool_call>` XML tags
- **Offline-first**: `LocalModelResolver` checks the sandboxed app container, system `~/.cache/huggingface/hub/`, and flat download layouts — no network requests if model is cached
- **No duplicate storage**: custom `HubApi(cache: nil)` with explicit `downloadBase` — models stored once in the snapshot cache, not duplicated across blob cache and snapshots
- **Thinking mode**: `enable_thinking` passed to Jinja template context via `additionalContext`; `<think>...</think>` tags parsed in real-time during streaming and shown in collapsible UI blocks. Toggleable in Settings.
- **Download progress**: separate `isDownloading` state from `isLoading`; modal overlay shows file count, percentage, speed
- **Idle unload**: timer resets on both user input and model generation completion (not just request start)
- **Chat export**: Markdown (user messages as blockquotes) and RTF (Pages-compatible with formatted markdown)
- **Finder paste**: local event monitor intercepts Cmd+V to check pasteboard for image file URLs before TextField handles it
- HTTP server built on `Network.framework` (`NWListener`) — no third-party server dependencies
- KV cache reuse across API requests — reuses `ChatSession` when conversation history prefix matches
- GPU cache limit set to 20 MB; cache cleared on model unload

## Dependencies

Managed via Swift Package Manager (declared in `project.yml` for xcodegen).

| Package | Products |
|---------|----------|
| `mlx-swift-lm` | `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| `swift-markdown-ui` | `MarkdownUI` |
