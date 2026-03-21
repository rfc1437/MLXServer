# Native Template Tool Formatting Plan

This document extracts Phase 7 item 19 from `session-cache-upgrade.md` into a standalone implementation plan.

The goal is to describe what would be required to move the API server from the current app-managed tool prompting approach to a model-template-native tool formatting approach later, without keeping the work buried inside the larger session/cache rewrite document.

## Summary

Current state:

- The app formats tool instructions itself.
- `PromptBuilder` injects tool definitions into prompt text.
- `ToolPromptBuilder` produces model-specific tool prompt text and replays assistant tool calls back into prompt history.
- `UserInput.tools` is currently not used for the API path.

Proposed future state:

- The app passes structured tools via `UserInput.tools`.
- The model's Jinja chat template formats tools natively.
- The app stops injecting tool instructions into the system prompt for models that are verified to support native template tools.
- Manual prompt formatting remains available as a fallback.

This is not a simple flag flip in the current codebase. It is a separate integration project.

## Why Consider This Later

Potential benefits:

- Less model-specific prompt text generation in app code.
- Closer alignment with template authors' intended tool formatting.
- Possible improvement in tool-call quality for models with reliable native tool templates.
- Reduced duplication between app-side prompt construction and template-side prompt construction.

Current reasons not to prioritize it immediately:

- The current manual path is already implemented and tested.
- Model-template behavior is not uniformly reliable. Phase 6 validation already showed that some local Qwen builds do not consistently honor their own documented thinking-tag contract.
- The current code does not yet contain a real runtime strategy switch between manual and native tool formatting.

## Current Implementation

Today, the API path does the following:

1. If tools are present, `PromptBuilder` appends a model-specific tool prompt into the instructions block.
2. Assistant tool calls in message history are rewritten back into model-native text form.
3. Tool outputs are also rewritten into model-specific history text.
4. `UserInput` is built with `tools: nil`.
5. Output parsing prefers framework-emitted tool calls first, then falls back to text parsing.

Files involved:

- `MLXServer/Server/PromptBuilder.swift`
- `MLXServer/Server/ToolPromptBuilder.swift`
- `MLXServer/Server/APIServer.swift`
- `MLXServer/Server/ToolCallParser.swift`

## Validated Local Model Templates

The following observations are based on the local model template files currently present in the MLX Server cache.

### Qwen3.5 0.8B, 4B, and 9B

Local Qwen3.5 templates do appear to support native tool formatting at the template level.

Observed capabilities in the local `chat_template.jinja` files:

- explicit `if tools` branch at the top of the template
- renders a `<tools>` block containing serialized tool definitions
- instructs the model to emit tool calls in a native Qwen XML format
- replays prior assistant `tool_calls` in template-native form
- replays `tool` role messages through `<tool_response>` wrappers

Implication:

- Qwen3.5 models are plausible candidates for a future `templateNative` allowlist.

Important caveat:

- template support on paper is not enough by itself. Phase 6 validation already showed that local Qwen3.5 builds do not consistently honor every documented template contract, specifically for `<think>...</think>` behavior. Native tool formatting for Qwen therefore still requires runtime validation, not just template inspection.

### Gemma 3 4B

The local Gemma template does not appear to support native tools.

Observed behavior in the local `chat_template.json`:

- no `tools` variable handling
- no native tool-definition rendering path
- no replay path for assistant `tool_calls`
- no dedicated `tool` role handling
- template structure is focused on alternating user/model turns and image placeholders only

Implication:

- Gemma must remain on the current manual prompt formatting path unless a different local template or upstream framework behavior is introduced.

### Practical Conclusion

If this work is taken on later, the initial allowlist should be:

- Qwen3.5 family: possible candidate, but only after runtime validation
- Gemma 3: not a candidate under the current local template

## Target Implementation

For verified models, the API path should be able to:

1. Convert OpenAI-format tool definitions into framework-native tool specs.
2. Pass those tool specs through `UserInput.tools`.
3. Avoid appending manual tool instructions to the system prompt.
4. Keep output parsing compatible with both framework-native tool call events and text fallback parsing.
5. Fall back to the current manual path when native template tool formatting is unsupported or broken.

## Impact On TokenPrefixCache And Prompt Reuse

This change does not require a redesign of `TokenPrefixCache`, but it does affect cache behavior and rollout strategy.

### 1. No Core Cache Algorithm Change Is Required

The current cache key is built from the prepared token sequence returned by `container.prepare(input:)`, plus image fingerprint augmentation for VL models.

That means:

- if tool formatting changes the rendered prompt, the token sequence changes
- if the token sequence changes, the cache key changes automatically
- prefix, supersequence, and LCP matching continue to work without algorithmic modification

So the cache implementation itself does not need a new matching strategy just for native-template tools.

### 2. Cache Hits Become Strategy-Sensitive

Even if the semantic request is identical, the manual path and the template-native path may render different prompt text.

Result:

- existing cache entries created under `manualPrompt` will usually not hit under `templateNative`
- this is expected and safe
- rollout will temporarily reduce cache hit rate for any model moved to the new path until fresh entries are built

There is no cache migration requirement. Old entries can simply age out.

### 3. Strategy Changes Can Fragment Cache Reuse

If the same model sometimes uses `manualPrompt` and sometimes uses `templateNative`, prompt reuse becomes less predictable because token prefixes will diverge.

Practical effect:

- more misses across otherwise similar requests
- less interpretable hit-rate statistics during rollout

Recommended mitigation:

- keep strategy stable per model
- use an explicit allowlist rather than opportunistic per-request switching

### 4. Deterministic Tool Serialization Matters More

TokenPrefixCache depends on byte-stable prompt rendering. If logically identical tool schemas are rendered with different key ordering or formatting across requests, cache hits will degrade.

This matters more under a native-template path because tool schema serialization moves closer to template/framework behavior.

Validation requirement:

- the same tool definitions must render to the same token sequence across runs for a stable cache key

This should be tested explicitly for any allowlisted model.

### 5. Multi-Turn Replay Has Direct Cache Impact

The current manual path reconstructs prior assistant tool calls and tool responses in deterministic model-specific text.

If the native-template path replays history differently, then:

- second-turn and later requests may produce different token prefixes
- prefix reuse depth may shrink
- supersequence and LCP opportunities may change even when conversation meaning is unchanged

So history replay semantics are not just a correctness concern; they also affect cache reuse quality.

### 6. Image-Aware Cache Keying Is Unchanged

The current vision cache-key augmentation based on image fingerprints is independent of tool formatting.

Implication:

- no change is needed to Gemma/Qwen image-aware cache key construction just because tools move from manual prompt text to `UserInput.tools`

### 7. Prompt Estimation May Need Adjustment

Today, `PromptBuilder` estimates prompt size before prepare using app-constructed instruction and message text.

Under a native-template path, some tool formatting moves inside the template/framework.

Impact:

- pre-prepare `estimatedBytes` and `estimatedPromptTokens` may become less representative
- the actual prepared token count remains authoritative for cache keys and post-prepare accounting

This does not break TokenPrefixCache, but it may require revisiting prompt estimation if UI or request validation depends on the earlier estimate.

## Recommended Design

### 1. Introduce a Real Strategy Type

Add an explicit strategy abstraction for the API path.

Suggested shape:

```swift
enum ToolFormattingStrategy {
    case manualPrompt
    case templateNative
}
```

This should become a real code path selector, not just a design note.

### 2. Do Not Auto-Detect Aggressively At First

The original note suggested auto-detecting whether a model template supports tools natively.

That is possible, but it is risky as an initial rollout because:

- preparation succeeding does not prove correct tool formatting
- a template may accept `tools` but produce malformed tool calls
- model behavior can still vary across quantized or repackaged local builds

Recommended first rollout:

- start with an explicit allowlist of models verified to work with native template tools
- keep all other models on the current manual path
- only add dynamic detection later if there is a clear need

### 3. Add Conversion From API Tools To Framework Tool Specs

`APIChatCompletionRequest.tools` uses the OpenAI-compatible app model.

To support template-native formatting, the app will need a conversion layer from:

- `APIToolDefinition`

to:

- the `mlx-swift-lm` native tool specification type used by `UserInput.tools`

Required work:

- map function names
- map descriptions
- map parameter schemas
- preserve required vs optional fields
- confirm how nested object/array schemas must be represented in the framework type

This conversion should live in a dedicated helper instead of being embedded directly inside `PromptBuilder`.

### 4. Update PromptBuilder To Support Both Paths

`PromptBuilder` currently always uses the manual path.

It will need to change so that:

- on `manualPrompt`, behavior stays the same as today
- on `templateNative`, manual system-prompt tool injection is skipped
- on `templateNative`, `UserInput.tools` is populated with converted tool specs

Important constraint:

- message-history handling for assistant tool calls and tool outputs may also need strategy-dependent treatment

The current replay logic assumes the app is responsible for reconstructing model-native text history. If the template-native path expects structured tool state instead, replay rules may need to change.

### 5. Verify History Replay Semantics

This is one of the main reasons item 19 is not a trivial switch.

Today, history replay is manual:

- assistant tool calls are converted back into Qwen `<tool_call>` or Gemma `tool_code`
- tool outputs are converted back into model-specific history text

Questions that must be answered for a native-template path:

1. Does the template expect previous assistant tool calls to appear as plain text, structured tool metadata, or both?
2. Does the template expect tool responses to be represented through normal chat messages only, or via another structured field?
3. Does the framework already shape those prior turns correctly when `UserInput.tools` is present?

If the answer is not fully consistent across models, the app will still need model-specific replay logic even under `templateNative`.

### 6. Keep Output Parsing Hierarchy As-Is

The output parsing hierarchy already matches the preferred design:

1. framework-emitted tool calls first
2. text parser fallback second

That part likely does not need architectural change.

However, the following should still be verified under the new path:

- non-streaming tool responses
- streaming tool-call chunks
- multi-turn tool conversations
- mixed content plus tool calls

### 7. Add Safe Fallback Behavior

This feature should not be all-or-nothing.

Recommended behavior:

- if model is not allowlisted, use `manualPrompt`
- if model is allowlisted but native template behavior fails validation, fall back to `manualPrompt`
- avoid silent partial activation

Possible rollout options:

- compile-time default to manual, enable native only in tests
- runtime flag for development builds
- per-model hardcoded allowlist after verification

## Suggested Implementation Steps

1. Add `ToolFormattingStrategy` and wire it through the API prompt-building path.
2. Add a converter from `APIToolDefinition` to framework-native tool specs.
3. Update `PromptBuilder` so `UserInput.tools` can be populated for the native path.
4. Keep manual prompt injection untouched as the fallback path.
5. Verify how prior assistant tool calls and tool outputs must be replayed for native-template mode.
6. Start with one verified model only.
7. Add end-to-end tests for that model.
8. Expand allowlist only after repeated validation.

## Testing Required

This work would require new focused tests beyond the current manual-path coverage.

Minimum required coverage:

- native-template tool path can prepare successfully with tools present
- model emits tool calls that the framework surfaces correctly
- non-streaming response returns `finish_reason == "tool_calls"` when appropriate
- streaming response emits OpenAI-compatible tool-call chunks in the correct order
- tool-call arguments survive round-trip without schema loss
- multi-turn tool conversation still replays correctly on the next request
- fallback to `manualPrompt` still works for models outside the allowlist

Recommended additional coverage:

- one test per supported native-template model
- explicit regression test for malformed tool output
- replay test with prior assistant tool calls plus tool responses in history

## Risks

Main risks:

- template behavior differs across local model builds
- framework-native tool support may accept a tool schema but not format prompts as expected
- replay semantics may still require model-specific handling, reducing the benefit of the switch
- debugging becomes harder because part of the prompt construction moves into model templates instead of app code

## Recommendation

Treat this as a future experiment, not pending polish.

It becomes worth doing only if at least one of these is true:

- the current manual tool path shows a real correctness bug
- a verified model demonstrates materially better tool behavior on the native-template path
- upstream framework support becomes stable and well-documented enough to reduce integration risk

Until then, the current manual implementation remains the safer default.