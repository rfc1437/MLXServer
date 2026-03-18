# Chat Document Format And Implementation Plan

## Recommendation

Use a custom macOS document package with the extension `.mlxchat`.

- User-facing behavior: appears as a single document in Finder and standard Open/Save panels.
- Internal structure: a file package directory containing a versioned JSON manifest plus binary attachments.
- Primary app integration: keep the current `WindowGroup` app structure initially, add explicit Open/Save/Revert flows, and only consider a later move to `DocumentGroup` if document-centric window management becomes a product goal.

This is the best fit for MLX Server because it needs to preserve more than transcript text:

- full message history
- pasted or dragged images
- assistant thinking blocks
- model and generation context
- future tool calls and tool results
- enough metadata to migrate old documents safely

## Why Not Other Formats

### Flat JSON file

Good for text-only chats. Weak once images and future binary artifacts need to travel with the conversation.

### ZIP container

Portable, but less native on macOS. Finder and AppKit document workflows work better with a package document.

### SQLite

Reasonable for an internal database, but a poor user-facing interchange format for a single chat document.

### Binary plist or `NSKeyedArchiver`

Too opaque and too brittle for a long-lived app document format.

## Proposed Package Layout

```text
Conversation.mlxchat/
  manifest.json
  attachments/
    9E8B7C7B-3F6E-4F77-8D17-0C7C0E8D5E62.png
    91A2A1D4-55C8-4D0D-BECB-5A2B5D276D8B.jpg
  previews/
    thumbnail.png
```

Notes:

- `manifest.json` is the canonical source of truth.
- `attachments/` contains lossless binary payloads referenced by manifest IDs.
- `previews/thumbnail.png` is optional and can be added later for Quick Look or Finder previews.

## UTType And Document Identity

Define a custom exported type.

- Identifier: `de.rfc1437.mlxserver.chat`
- Conforms to: `com.apple.package`, `public.data`
- Filename extension: `mlxchat`
- Display name: `MLX Server Chat Document`

In this project, that will require updates in [project.yml](project.yml) so the generated Info.plist exports the document type.

## Manifest Schema

The manifest should stay plain JSON and be versioned from day one.

### Top-level shape

```json
{
  "schemaVersion": 1,
  "documentId": "1F8A10B6-7D1B-4D2B-9D3B-18B9818D3C17",
  "createdAt": "2026-03-18T12:00:00Z",
  "updatedAt": "2026-03-18T12:34:56Z",
  "appVersion": "1.0.0",
  "model": {
    "id": "qwen3.5-9b",
    "displayName": "Qwen3.5 9B",
    "repoId": "mlx-community/Qwen3.5-9B-4bit"
  },
  "settings": {
    "systemPrompt": "",
    "thinkingEnabled": true,
    "temperature": 0.7
  },
  "messages": [],
  "toolEvents": [],
  "uiState": {
    "draftInput": "",
    "scrollAnchorMessageId": null
  }
}
```

### Message shape

```json
{
  "id": "7A0D764B-D40D-4A7E-9655-441EF79C2B44",
  "role": "assistant",
  "createdAt": "2026-03-18T12:01:00Z",
  "content": "Visible response text",
  "rawContent": "<think>reasoning</think>Visible response text",
  "thinkingContent": "reasoning",
  "streamingState": "completed",
  "attachments": [
    {
      "id": "9E8B7C7B-3F6E-4F77-8D17-0C7C0E8D5E62",
      "type": "image",
      "relativePath": "attachments/9E8B7C7B-3F6E-4F77-8D17-0C7C0E8D5E62.png",
      "mimeType": "image/png",
      "pixelWidth": 1024,
      "pixelHeight": 768,
      "sha256": "..."
    }
  ]
}
```

### Schema rules

- Never serialize `NSImage` directly.
- Store timestamps in ISO 8601 UTC.
- Store stable UUIDs for document, messages, and attachments.
- Treat `rawContent` as canonical for assistant messages so future parsing changes do not destroy original output.
- Keep `thinkingContent` denormalized for fast restore and forward compatibility.
- Ignore unknown keys on read so older app versions degrade gracefully.

## State To Preserve

Minimum state required for a true restore:

- all `Conversation.messages`
- message timestamps
- user-attached images
- assistant `rawContent`, `content`, `thinkingContent`, and streaming completion state
- selected model identity at the time of the chat
- relevant generation settings used to produce the conversation
- current system prompt

State that should not be persisted initially:

- active `ChatSession` or KV cache
- live token counters
- running API server state
- in-flight generation tasks

Restored chats should reproduce visible history, not attempt to resume an MLX runtime session.

## Required Refactor Before Saving

The current chat model is UI-oriented and not suitable as the storage contract.

Relevant existing code:

- [MLXServer/Models/ChatMessage.swift](MLXServer/Models/ChatMessage.swift)
- [MLXServer/ViewModels/ChatViewModel.swift](MLXServer/ViewModels/ChatViewModel.swift)
- [MLXServer/Utilities/ChatExporter.swift](MLXServer/Utilities/ChatExporter.swift)
- [MLXServer/ContentView.swift](MLXServer/ContentView.swift)

### Problem areas in the current model

- `ChatMessage.id` is regenerated on every launch because it is not decoded from persisted data.
- `timestamp` is assigned in the initializer and cannot currently be restored.
- `images` uses `NSImage`, which is a view/runtime type rather than a storage type.
- `Conversation` has no import/export path other than Markdown and RTF export.

### Recommended split

Introduce two distinct layers:

1. Persistent document model
2. Runtime/UI model

Suggested types:

- `ChatDocumentPackage`: `FileDocument` implementation for `.mlxchat`
- `ChatDocumentManifest`: Codable root object
- `StoredChatMessage`: Codable message payload
- `StoredAttachment`: Codable attachment metadata
- `ChatAttachment`: runtime representation with file URL and lazily loaded image

This keeps the JSON schema stable while allowing UI and MLX runtime concerns to evolve independently.

## Swift API Design

### New files

- `MLXServer/Documents/ChatDocumentPackage.swift`
- `MLXServer/Documents/ChatDocumentManifest.swift`
- `MLXServer/Documents/ChatDocumentMigration.swift`

### Adjust existing files

- [MLXServer/Models/ChatMessage.swift](MLXServer/Models/ChatMessage.swift): make the runtime model restorable from stored values and stop using implicit-only timestamps for persisted messages.
- [MLXServer/ViewModels/ChatViewModel.swift](MLXServer/ViewModels/ChatViewModel.swift): add load/save hooks and dirty-state tracking.
- [MLXServer/ContentView.swift](MLXServer/ContentView.swift): add open/save/save-as/revert/import wiring.
- [MLXServer/Commands/SaveChatCommands.swift](MLXServer/Commands/SaveChatCommands.swift): expand from export-only commands to proper document commands.
- [project.yml](project.yml): register exported document type and include new source files.

### `FileDocument` responsibilities

Write path:

- serialize manifest JSON
- encode attachments into deterministic filenames
- build package directory with `FileWrapper(directoryWithFileWrappers:)`

Read path:

- validate package structure
- decode manifest
- resolve referenced attachments
- migrate old schema versions before handing data to the app

## Integration Strategy

## Phase 1: Add document support without changing the app architecture

Recommended first implementation.

- Keep `MLXServerApp` as a `WindowGroup` app.
- Add `Open Chat…`, `Save Chat…`, `Save Chat As…`, and `Revert To Saved` commands.
- Use `fileImporter` and `fileExporter` from the current main window.
- Track the current document URL and dirty state inside `ChatViewModel`.

Why this first:

- minimal disruption to model loading and existing window behavior
- faster path to a working restore feature
- avoids forcing document-window semantics into the app before the document model is proven

### Required view model additions

Add to `ChatViewModel`:

- `currentDocumentURL: URL?`
- `hasUnsavedChanges: Bool`
- `lastSavedSnapshotHash: String?`
- `loadDocument(from:)`
- `saveDocument(to:)`
- `markDirtyIfNeeded()`

Mark the conversation dirty when:

- a message is added
- an attachment is added or removed
- streamed assistant output is finalized
- document-level settings that belong in the manifest change

## Phase 2: Consider migration to `DocumentGroup`

Optional and only worth doing if you want standard multi-document macOS behavior.

Benefits:

- native new/open/reopen document lifecycle
- automatic recent documents support
- better fit for multiple chat windows

Costs:

- larger refactor to inject `ModelManager` into document scenes cleanly
- more work to preserve current app-global settings and API server assumptions

Recommendation: do not start here.

## Save Semantics

Use standard desktop-document behavior.

- `Save`: overwrite current `.mlxchat` if the chat already has a document URL.
- `Save As`: always prompt for a new destination.
- `Open`: replace the current conversation after unsaved-changes confirmation.
- `New Chat`: if dirty, prompt before clearing.
- autosave: optional second step after manual save/open works reliably.

Unsaved change detection can be implemented by hashing a canonical serialized manifest without volatile fields such as `updatedAt`.

## Attachment Handling

Store images as files in `attachments/`, not inline base64 JSON.

Recommended rules:

- Prefer PNG for screenshots and lossless clipboard images.
- Preserve original file format when importing from disk if feasible.
- Compute SHA-256 for deduplication and integrity checks.
- Do not rely on original absolute file paths after import.
- Restore images from document-local paths only.

Runtime API suggestion:

- `ChatAttachment.imageURL`
- `ChatAttachment.loadNSImage()`

This avoids embedding heavy `NSImage` values in the persistence model.

## Migration Strategy

Implement migrations immediately, even with only one schema version.

- `schemaVersion = 1` for the first release
- central migration entry point that upgrades any older manifest to current model structs
- log and surface a clear user error if a future version is unsupported

This prevents format lock-in and avoids dangerous ad hoc decode logic later.

## Error Handling

User-facing errors should distinguish between:

- invalid document package
- missing manifest
- corrupted attachment reference
- unsupported schema version
- failed attachment decode

Do not partially mutate the current conversation until a full document read succeeds.

## Testing Plan

Add focused tests for document round-tripping.

### Unit tests

- manifest encode/decode round trip
- message round trip with thinking content
- attachment reference round trip
- migration from schema version 1 fixture
- dirty-state hash stability

### Integration tests

- save a chat with images, reopen it, and verify visible conversation equality
- save a chat with assistant thinking blocks, reopen it, and verify both visible and thinking content
- open a malformed package and verify the app shows a recoverable error

### Manual verification

- create a new chat, save as `.mlxchat`, quit app, reopen document, verify full restore
- duplicate the file in Finder and open the copy
- rename the file in Finder and reopen
- test a large image attachment document

## Implementation Sequence

1. Introduce Codable stored document types and a versioned manifest.
2. Refactor runtime chat state so messages and attachments can be reconstructed from stored data.
3. Implement `.mlxchat` package read/write using `FileDocument`.
4. Register UTType and document type metadata in [project.yml](project.yml).
5. Add open/save/save-as/revert commands and wire them into [MLXServer/ContentView.swift](MLXServer/ContentView.swift) and [MLXServer/Commands/SaveChatCommands.swift](MLXServer/Commands/SaveChatCommands.swift).
6. Add unsaved-change prompts around new/open flows.
7. Keep Markdown/RTF export as a separate export feature, not the primary restore format.
8. Add round-trip tests and a small set of fixture documents.

## Concrete Recommendation For This Codebase

Implement `.mlxchat` as a package document now, but keep the current `WindowGroup` app structure for the first pass.

That gives MLX Server a native macOS document format without taking on an unnecessary app-architecture rewrite. Once the format and save/open behavior are stable, reassess whether the product would benefit from a full `DocumentGroup` migration.