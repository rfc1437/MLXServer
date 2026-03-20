# Session & Cache Upgrade: Stateless API Server with Token-Level Prefix Caching

This document specifies a full rewrite of MLX Server's API server inference path. The goal is to drop `ChatSession` for the API path entirely, go fully stateless like vllm-mlx, and build directly on `mlx-swift-lm`'s lower-level APIs (`ModelContainer`, `generate()`, `[KVCache]`, `UserInputProcessor`, `Tokenizer`). This gives us direct KV cache control for token-level prefix caching.

The UI chat path (`ChatViewModel.swift`) keeps using `ChatSession` — it's a different use case and works fine there.

All code is Swift. The project uses `mlx-swift-lm` (`MLXVLM` / `VLMModelFactory`) as its inference backend.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Stateless Inference Engine](#2-stateless-inference-engine)
3. [Vision-Language Model Support](#3-vision-language-model-support)
4. [Token-Level Prefix Cache](#4-token-level-prefix-cache)
5. [Prompt Builder](#5-prompt-builder)
6. [Native Template Tool Formatting](#6-native-template-tool-formatting)
7. [Client Disconnect Detection](#7-client-disconnect-detection)
8. [Optimized SSE Encoder](#8-optimized-sse-encoder)
9. [Qwen3 EOS Token Fix](#9-qwen3-eos-token-fix)
10. [APIServer Rewrite](#10-apiserver-rewrite)
11. [Statistics & Monitoring Upgrade](#11-statistics--monitoring-upgrade)
12. [Advanced Cache Matching](#12-advanced-cache-matching)
13. [KV Cache Quantization](#13-kv-cache-quantization)
14. [File-by-File Change Map](#14-file-by-file-change-map)
15. [Implementation Order](#15-implementation-order)
16. [Testing Checklist](#16-testing-checklist)

---

## 1. Architecture Overview

### Current Architecture (ChatSession-based)

```
API Request
  → Convert messages to Chat.Message
  → Match session by message-hash signatures
  → Hit: reuse ChatSession (opaque KV cache inside)
  → Miss: create new ChatSession with history replay
  → ChatSession.streamDetails() — applies template, manages cache internally
  → Stream tokens back as SSE
```

**Problems:**
- No access to KV cache — can't trim, fork, or inspect it.
- Message-level matching (FNV-1a hash) — one character change invalidates everything.
- +1 message constraint — tool-use flows with 2+ new messages always miss.
- ChatSession applies the chat template internally — we can't inspect or cache the tokenized prompt.
- No way to reuse partial prefill across different conversations that share a system prompt.

### New Architecture (Stateless, Direct API)

```
API Request
  → Build prompt via UserInputProcessor.prepare()
  → Tokenize to get [Int] token array
  → Look up token prefix in TokenPrefixCache → get [KVCache] + matchedCount
  → If partial match: trim KV cache to matchedCount, prefill only remaining tokens
  → Call generate(input:, cache:, parameters:, context:) directly
  → Stream tokens back as SSE
  → Store [KVCache] + full token array back in TokenPrefixCache
```

**Wins:**
- Token-level prefix reuse — shared system prompts, partial conversation matches, everything works.
- No message-count constraints — any prefix match works.
- Direct KV cache management — trim, clone, evict by memory pressure.
- Template applied once, tokenized once, cached as token array.
- Fully stateless per request — no session objects to manage.

### Out of Scope

The following are explicitly **not covered** in this revision:

- **Request scheduling / concurrency limits / admission control.** `ModelContainer.perform()` serializes all model access via its internal lock — concurrent API requests queue implicitly. For a local single-user server, this implicit FIFO queue is adequate: requests are mostly sequential (send message → wait for response → send next), and even in agentic scenarios with 2-3 parallel tool calls, the serial queue ensures each request gets full GPU bandwidth. A priority queue or rejection policy would matter for multi-client deployment where you need fairness guarantees or want to shed load under pressure, but that's not the current use case.

- **SSE backpressure.** `NWConnection.send()` buffers internally if a slow client can't consume events fast enough. In practice this is a non-issue for a local server: the client is on the same machine with negligible network latency, so send buffers don't grow. For a remote-serving scenario you'd want explicit flow control (pause generation if send buffer exceeds a threshold) or client eviction for stalled connections, but local clients consume tokens faster than they're generated.

- **Batch inference / continuous batching.** Each request runs a single `generate()` call. vllm-mlx batches multiple sequences into one forward pass to maximize GPU utilization across concurrent users. For MLX Server this has low value for two reasons: (1) `ModelContainer.perform()` takes an exclusive serial lock, so batching would require upstream API changes to mlx-swift-lm (a batch-aware generate accepting multiple inputs/caches). (2) On Apple Silicon, generation is memory-bandwidth-bound, not compute-bound — batching multiple sequences doesn't yield the same throughput multiplier as on NVIDIA GPUs where compute parallelism is underutilized by a single sequence. The one scenario where batching would help is overlapping prefill of one request with generation of another, but that's a niche win for a local single-user app that doesn't justify the upstream API dependency. Worth revisiting if MLX Server ever targets multi-client serving.

### Key mlx-swift-lm APIs Used

| API | Purpose |
|-----|---------|
| `ModelContainer.processor` | Get the `UserInputProcessor` for prompt preparation |
| `ModelContainer.tokenizer` | Direct tokenizer access for encode/decode |
| `ModelContainer.prepare(input:)` | Convert `UserInput` → `LMInput` (thread-safe convenience) |
| `ModelContainer.perform { ctx in ... }` | Access `ModelContext` inside serial lock — required for cache-aware generation |
| `MLXLMCommon.generate(input:cache:parameters:context:)` | **Module-level** function that streams `Generation` events with explicit `[KVCache]?`. Called inside `perform`. |
| `model.newCache(parameters:)` | Create fresh `[KVCache]` |
| `KVCache.offset` | Current token count in cache |
| `KVCache.state` / `KVCache.metaState` | Serializable KV state (get/set) |
| `KVCache.isTrimmable` | Whether the cache supports trimming |
| `KVCache.trim(_:)` | Trim N tokens from the cache, returns actual count trimmed |
| `UserInputProcessor.prepare(input:)` | Convert messages + images → `LMInput` |
| `GenerateParameters` | Temperature, topP, maxTokens, prefillStepSize, kvBits, etc. |
| `Generation.chunk/info/toolCall` | Streaming output events |

> **API clarification (verified against mlx-swift-lm source):**
>
> `ModelContainer.generate(input:parameters:)` does **not** accept a `cache:` parameter — it always creates a fresh `[KVCache]` internally. To pass an explicit cache, you must use `container.perform { context in ... }` to get a `ModelContext`, then call the **module-level** `MLXLMCommon.generate(input:cache:parameters:context:)` which does accept `cache: [KVCache]?`.
>
> This is the same pattern `ChatSession` uses internally — it holds `[KVCache]` in a private `SerialAccessContainer<Cache>` and calls the module-level function inside `perform`.
>
> **Sendability constraint:** `[KVCache]` contains `MLXArray` which is not `Sendable`. The cache array cannot cross isolation boundaries directly. `TokenPrefixCache` must use `@unchecked Sendable` (protected by its own `OSAllocatedUnfairLock`) and all cache access must happen either inside `perform` or under the cache's lock.
>
> **Cache mutability model:** `KVCache` is a reference type (class). The `generate()` function mutates cache objects in place during generation — it calls `update(keys:values:)` on each layer to append new K/V states. This means a cache handed out from `TokenPrefixCache` will be modified by the generation process. The cache must therefore be **checked out** (removed from the trie) on lookup, not merely borrowed. After generation, the mutated cache is stored back at its new (longer) key. See the Cache Lease Lifecycle section below for the full protocol.
>
> vllm-mlx's "no deep copies on fetch: MLX arrays are immutable" applies to the individual `MLXArray` tensors within a KV layer (which are copy-on-write value semantics at the MLX level), NOT to the `KVCache` object itself which is mutated by `generate()`.

---

### Cache Lease Lifecycle

The KV cache flows through a **checkout → generate → store** lifecycle. This is the central correctness invariant of the new architecture.

```
1. LOOKUP (checkout)
   TokenPrefixCache.lookup(cacheKey, modelId)
     → Hit:  REMOVE the entry from the trie. Return the [KVCache] + matchedTokenCount.
             The entry is consumed, not borrowed — the trie no longer references it.
     → Miss: Return nil cache. Caller creates a fresh cache via model.newCache().

2. CREATE WORKING CACHE
   → Hit:  The checked-out [KVCache] IS the working cache. Its offset == matchedTokenCount.
           generate() will mutate it in place (append K/V states for new tokens).
   → Miss: Create fresh cache inside container.perform { context in
               context.model.newCache(parameters: ...)
           }
           This is the working cache for the cold-start path.

3. GENERATE (mutates working cache in place)
   MLXLMCommon.generate(input: fullLMInput, cache: workingCache, ...)
     → The framework calls cache.update(keys:values:) per layer per prefill step
       and per generated token. workingCache.offset grows as generation proceeds.
     → On cancellation: stop consuming the stream. workingCache contains valid
       state up to whatever was generated — still usable for storage.

4. STORE (insert at new key)
   After generation completes (or on cancellation):
     → Trim oversized KV arrays to actual offset (trimCacheToOffset)
     → Store at the PROMPT key only: prepResult.cacheKey
       (Do NOT append generated token IDs — see note below)
     → The trie now has an entry at the prompt key covering prompt tokens.
       The next request's prompt will include prior assistant text, so
       prefix matching works naturally without needing generated token IDs.

5. ABANDON (on error)
   If generation fails, the working cache is simply dropped (no store).
   Since the entry was removed from the trie at checkout, there's nothing to clean up.
```

**Why checkout removes the entry:** `KVCache` is a reference type. If `lookup()` returned a reference while keeping the entry in the trie, `generate()` would mutate the stored entry in place — corrupting it for any future lookup that expects the original (shorter) prefix. By removing the entry on checkout, we guarantee exclusive ownership during generation.

**Why store uses only prompt tokens as key:** The plan originally appended `generatedTokenIds` to the cache key. This is unnecessary and problematic:
- The `Generation` stream from `MLXLMCommon.generate()` yields `.chunk(String)` (decoded text), not raw token IDs. Recovering exact IDs would require re-tokenization, which may not round-trip identically for all text (special tokens, tool-call formatting).
- It's also not needed: the next API request's prompt naturally includes the prior assistant response as a message. `PromptBuilder` + `UserInputProcessor.prepare()` tokenizes the full conversation, producing a token array that starts with the same prefix. The trie match covers the shared prefix automatically.

**Concurrency note:** `ModelContainer.perform()` serializes all access to the model. Only one `generate()` call runs at a time. Combined with checkout-removes-entry, there is no concurrent mutation of the same `[KVCache]` objects.

---

## 2. Stateless Inference Engine

### New Type: `InferenceEngine`

Create file: `MLXServer/Server/InferenceEngine.swift`

This replaces all `ChatSession` usage in the API path. It is a stateless request handler — each call takes a fully prepared prompt plus an optional cached KV state, runs generation, and returns results.

```swift
import MLX
import MLXLMCommon
import MLXVLM

/// Stateless inference engine for the API server.
/// Each request gets its own generate() call with explicit KV cache management.
/// Thread safety is handled by ModelContainer's internal serialization.
///
/// **API pattern:** ModelContainer.generate() does NOT accept a cache: parameter.
/// To pass explicit [KVCache], we use container.perform { context in ... } to get
/// a ModelContext, then call the module-level MLXLMCommon.generate(input:cache:parameters:context:).
/// This is the same pattern ChatSession uses internally.
final class InferenceEngine: @unchecked Sendable {

    private let container: ModelContainer

    init(container: ModelContainer) {
        self.container = container
    }

    // MARK: - Public API

    /// Request parameters for inference. Note: LMInput and [KVCache] are NOT Sendable
    /// (they contain MLXArray). This struct is only constructed and consumed inside
    /// perform() or under the cache lock — never sent across isolation boundaries.
    struct InferenceRequest {
        let input: LMInput
        let tokens: [Int]               // full tokenized prompt (for cache keying)
        let parameters: GenerateParameters
        let cachedKV: [KVCache]?         // nil = cold start; non-nil = checked-out prefix hit
        let cachedTokenCount: Int        // how many tokens the cached KV covers
    }

    /// Result of starting a streaming inference. Bundles the generation stream
    /// with the working KV cache so the caller can store it after consumption.
    struct StreamHandle {
        let stream: AsyncStream<Generation>
        /// The working [KVCache] being mutated by generate().
        /// On cache hit: this is the checked-out cache from TokenPrefixCache.
        /// On cold start: this is a freshly created cache from model.newCache().
        /// After the stream is fully consumed, this cache contains K/V states
        /// for the full prompt + all generated tokens. Trim and store it.
        let workingCache: [KVCache]
    }

    /// Run a streaming inference with explicit KV cache.
    ///
    /// Uses `container.perform { context in ... }` to access the ModelContext,
    /// then calls the module-level `MLXLMCommon.generate(input:cache:parameters:context:)`
    /// which accepts an explicit `cache: [KVCache]?` parameter.
    ///
    /// Returns a StreamHandle containing both the generation stream AND the
    /// working [KVCache]. The caller consumes the stream, then trims and stores
    /// the working cache via TokenPrefixCache. See Cache Lease Lifecycle.
    func stream(
        _ request: InferenceRequest,
        cancellation: CancellationToken
    ) async throws -> StreamHandle {
        // Transfer non-Sendable values into the perform closure.
        // Safe: we consume them exactly once inside the closure.
        nonisolated(unsafe) let input = request.input
        nonisolated(unsafe) let cachedKV = request.cachedKV
        let parameters = request.parameters

        return try await container.perform { context in
            // Cold start: create fresh cache. Hit: reuse checked-out cache.
            // We must always pass a non-nil cache so we retain a reference
            // to the object that generate() mutates in place.
            let workingCache = cachedKV ?? context.model.newCache(parameters: parameters)

            let stream = try MLXLMCommon.generate(
                input: input,
                cache: workingCache,
                parameters: parameters,
                context: context
            )

            return StreamHandle(stream: stream, workingCache: workingCache)
        }
    }

    /// Prepare a UserInput into an LMInput, and return the token array.
    /// Uses ModelContainer.prepare() which is a thread-safe convenience method.
    func prepare(_ userInput: UserInput) async throws -> PreparedInference {
        // prepare() is a convenience on ModelContainer — no need for perform()
        let lmInput = try await container.prepare(input: userInput)

        // Extract token array for cache keying.
        // LMInput.text.tokens is an MLXArray — convert to [Int].
        // This must happen inside perform() because MLXArray isn't Sendable,
        // but prepare() already evaluated the tokens, so asArray is safe here.
        let tokenArray: [Int] = await container.perform { context in
            let arr = lmInput.text.tokens.asArray(Int.self)
            return arr
        }

        return PreparedInference(
            lmInput: lmInput,
            tokens: tokenArray,
            hasImages: userInput.images.count > 0
        )
    }

    struct PreparedInference {
        let lmInput: LMInput       // full input including .image/.video
        let tokens: [Int]          // raw token array (for cache keying)
        let hasImages: Bool        // whether vision processing is involved
    }

    /// Get the tokenizer for direct token operations.
    var tokenizer: Tokenizer {
        get async { await container.tokenizer }
    }
}
```

### Key Design Points

1. **No state between requests.** Each call to `stream()` is independent. The KV cache follows a checkout → generate → store lifecycle (see Cache Lease Lifecycle above). `stream()` returns a `StreamHandle` containing both the generation stream and the working `[KVCache]` reference. On cold start, the cache is created inside `perform()` via `model.newCache()`. On cache hit, the checked-out cache is passed through. In both cases, the working cache is mutated in place by `generate()` and available to the caller for trimming and storage after the stream is consumed.

2. **Cache is the caller's responsibility.** `APIServer` calls `TokenPrefixCache.lookup()` (which checks out and removes the entry from the trie) before calling `stream()`, and `TokenPrefixCache.store()` (which inserts at the prompt key) after the stream is consumed.

3. **Cancellation is cooperative.** The `CancellationToken` is checked per-token in the consuming loop (not inside the stream producer). When cancelled, the consumer stops iterating and the stream drains naturally — we do NOT call `task.cancel()` on the underlying Metal work to avoid assertion failures (same lesson as vllm-mlx).

4. **ModelContainer.perform() is the bridge.** `ModelContainer.generate()` doesn't expose `cache:`. We use `container.perform { context in MLXLMCommon.generate(input:cache:parameters:context:) }` to get the `ModelContext` and call the module-level function directly. This holds the container's serial lock during prefill only — the same behavior as the convenience method.

5. **Non-Sendable types require careful handling.** `LMInput`, `[KVCache]`, and `MLXArray` are not `Sendable`. They must be transferred into `perform` closures via `nonisolated(unsafe)` (safe because we consume them exactly once). `TokenPrefixCache` stores `[KVCache]` under its own `OSAllocatedUnfairLock` with `@unchecked Sendable` — the same pattern `ChatSession` uses internally with `SerialAccessContainer<Cache>`.

6. **KV arrays should be trimmed before storage.** Following vllm-mlx's `_trim_to_offset()` pattern: KV arrays are often pre-allocated larger than needed (e.g., 4096 slots when only 100 are used). Before storing in `TokenPrefixCache`, slice K/V tensors to their actual `offset` and `mx.eval()` the result so the original large buffer can be freed. Without this, cached entries waste memory on empty pre-allocated space.

---

## 3. Vision-Language Model Support

Vision is a primary feature. VL models (Gemma 3, Qwen3-VL) process images through a vision encoder that runs during `model.prepare()`, merging vision features with text embeddings before the language model runs. This has critical implications for the stateless architecture and KV cache reuse.

### How VL Models Work Internally

```
LMInput (from UserInputProcessor.prepare())
├── text.tokens: [BOS, <img_placeholder>, "Describe", "this"]
├── image.pixels: MLXArray [1, 3, 1344, 1344]   ← raw pixel data
└── image.frames: [THW(1, 1344, 1344)]           ← spatial metadata

    ↓ model.prepare(input, cache, windowSize)

1. visionModel(pixels, gridTHW) → visionFeatures [1, 256, hidden_dim]
   (256 patch tokens from a 1344x1344 image)

2. textEmbeddings = languageModel.embedTokens(text.tokens)

3. mergedEmbeddings = replace <img_placeholder> with visionFeatures
   Result: [BOS_embed, vision_0, vision_1, ..., vision_255, "Describe", "this"]

4. languageModel(inputIds, inputsEmbeds: mergedEmbeddings, cache: kvCache)
   → KV cache now contains attention states for the full merged sequence

    ↓ subsequent generation tokens

5. model.callAsFunction(prevToken, cache: kvCache)
   → pixelValues: nil  (vision model NOT re-invoked during generation)
   → pure autoregressive text generation using cached attention states
```

### KV Cache Reuse Rules for VL Models

**What the KV cache contains:** Language model attention states (K/V tensors per layer) computed from the merged text+vision embedding sequence.

**What the KV cache does NOT contain:** Raw pixel data, vision encoder features, or any reference to the original image.

This means:

| Scenario | Cache Reusable? | Why |
|----------|----------------|-----|
| Same text, same images, append new text | **Yes** | KV states for the merged prefix are identical |
| Same text, different images | **No** | Vision features differ → different merged embeddings → different KV states |
| Same text, no images → same text, now with image | **No** | Token sequence changes (image placeholder tokens added) |
| Text-only prefix shared across conversations | **Yes** | No vision features involved in the shared prefix |
| Same image, different text after image | **Partially** | Tokens up through the image are reusable |

### Cache Key Design for VL Models

The token array alone is insufficient for cache keying with VL models. Two requests with identical token sequences but different images would produce different KV states. The cache key must include image identity.

#### Image Fingerprinting

```swift
/// Compute a fast fingerprint of an image for cache key purposes.
/// Does NOT hash every pixel — uses dimensions + sampled pixels for speed.
enum ImageFingerprint {
    static func compute(_ image: UserInput.Image) -> UInt64 {
        // Convert to a consistent representation
        // Use: width, height, and a sparse sample of pixel values
        // FNV-1a hash of: "\(width)x\(height):\(sample_pixels)"
        //
        // For data-URI inputs (base64), we can hash the raw base64 string
        // directly — it's already a unique identifier for the image data.
        var hash: UInt64 = 14_695_981_039_346_656_037
        // ... hash dimensions + sparse pixel sample
        return hash
    }
}
```

For base64 data-URI images (the common case in API requests), hash the raw base64 string before decoding — this is a free unique identifier that's already available.

#### Extended Cache Key

The cache trie key becomes `[Int]` where the array contains both token IDs and image fingerprint sentinels:

```swift
/// Build a cache key that incorporates both tokens and image identity.
/// Image fingerprints are injected at the positions where image placeholder
/// tokens appear in the token sequence.
static func buildCacheKey(
    tokens: [Int],
    imageFingerprints: [UInt64],
    imageTokenId: Int  // e.g., 248056 for Qwen3-VL's <image> token
) -> [Int] {
    var key: [Int] = []
    var imageIdx = 0

    for token in tokens {
        key.append(token)
        if token == imageTokenId && imageIdx < imageFingerprints.count {
            // Inject fingerprint as two sentinel values after the image token.
            // Use negative values (impossible as real token IDs) to avoid collisions.
            let fp = imageFingerprints[imageIdx]
            key.append(-Int(fp >> 32) - 1)        // high 32 bits (negated)
            key.append(-Int(fp & 0xFFFFFFFF) - 1)  // low 32 bits (negated)
            imageIdx += 1
        }
    }

    return key
}
```

This ensures that:
- Text-only prompts use pure token keys (no overhead).
- Prompts with images get unique keys per image.
- The trie naturally handles prefix sharing up to the first divergent image.

### Passing Images Through the Stateless Pipeline

#### InferenceEngine Changes

The `prepare()` method must preserve the full `LMInput` including `.image` and `.video` fields:

```swift
func prepare(_ userInput: UserInput) async throws -> PreparedInference {
    let lmInput = try await container.prepare(input: userInput)

    // Extract token array for cache keying
    let tokens = lmInput.text.tokens.asArray(Int.self)

    // Detect image placeholder token ID from the tokenizer
    let imageTokenId = await detectImageTokenId()

    // Build extended cache key with image fingerprints
    let imageFingerprints = userInput.images.map { ImageFingerprint.compute($0) }
    let cacheKey = Self.buildCacheKey(
        tokens: tokens,
        imageFingerprints: imageFingerprints,
        imageTokenId: imageTokenId
    )

    return PreparedInference(
        lmInput: lmInput,       // FULL input — includes .image/.video
        tokens: tokens,          // raw token array
        cacheKey: cacheKey,      // extended key with image fingerprints
        hasImages: !userInput.images.isEmpty
    )
}
```

#### Critical: Always Pass Full LMInput on Cache Hit

Even when the KV cache is reused, the `generate()` call receives the full `LMInput` with image data. The framework's `model.prepare()` checks `cache[0].offset` vs `input.text.tokens.count`:

- If `cache.offset == tokens.count`: all tokens cached, skip directly to generation.
- If `cache.offset < tokens.count`: only prefill the delta tokens.

For VL models with cached KV state, `cache.offset` already covers the image tokens, so `model.prepare()` skips the vision encoder entirely — it only runs for the uncached text suffix. The image pixels in `LMInput` are unused on cache hit (but must still be present because the framework's type system requires them).

```swift
// Cache hit: pass full LMInput + cached KV via perform()
// The framework sees cache.offset covers the image region → skips visionModel
let stream = try await container.perform { context in
    try MLXLMCommon.generate(
        input: prepared.lmInput,    // includes .image (but won't be re-processed)
        cache: lease.kvCache,       // already contains vision+text attention states
        parameters: params,
        context: context
    )
}
```

#### Edge Case: Images in Non-Final Messages

When images appear in earlier messages (not the last one), the KV cache from a previous request already covers those image tokens. The trie lookup matches on the extended cache key (which includes image fingerprints), so:

- Same images in same positions → cache hit, vision encoder skipped.
- Different image in an earlier message → cache miss from that point onward, vision encoder runs for the new image during full prefill.

### PromptBuilder Changes for VL Models

`PromptBuilder` must:

1. Attach images to individual `Chat.Message` objects (not a flat global list). VL models' templates expect images bound to specific messages.

2. Preserve image order matching the message order — the tokenizer inserts `<image>` placeholder tokens at positions corresponding to each message's images.

3. Collect image fingerprints alongside messages for cache key building.

```swift
// In PromptBuilder.build():
for msg in request.messages where msg.role != "system" {
    // ... text processing ...

    let imageURLs = msg.content?.imageURLs ?? []
    var messageImages: [UserInput.Image] = []
    var messageFingerprints: [UInt64] = []

    for urlString in imageURLs {
        // Hash the raw base64 string BEFORE decoding (free fingerprint)
        let fingerprint = ImageFingerprint.computeFromBase64(urlString)
        messageFingerprints.append(fingerprint)

        if let decoded = ImageDecoder.decode(urlString) {
            messageImages.append(decoded.image)
        }
    }

    // Images attached to THIS message, not globally
    chatMessages.append(Chat.Message(
        role: role,
        content: text,
        images: messageImages
    ))

    allFingerprints.append(contentsOf: messageFingerprints)
}
```

### VLM-Specific Testing Requirements

- [ ] Single image + text prompt → correct vision processing → coherent response
- [ ] Multi-image message → all images processed
- [ ] Image in message 1, text-only message 2 → cache reuse on message 3
- [ ] Same conversation, same image repeated → cache hit (vision encoder skipped)
- [ ] Same conversation, different image → cache miss, fresh vision processing
- [ ] Text-only conversation with VL model → no vision overhead, normal cache behavior
- [ ] Large images (4K+) → proper resize by UserInputProcessor, no OOM
- [ ] Mixed: image in user message, then assistant response, then user text-only follow-up → cache hit covers everything through the assistant response

---

## 4. Token-Level Prefix Cache

### New Type: `TokenPrefixCache`

Create file: `MLXServer/Server/TokenPrefixCache.swift`

This replaces `ConversationSessionCache` entirely. It stores `[KVCache]` arrays keyed by token sequences in a trie, with memory-aware eviction.

```swift
import Foundation
import Metal
import MLXLMCommon
import os

final class TokenPrefixCache: @unchecked Sendable {
    static let shared = TokenPrefixCache()

    private let lock = OSAllocatedUnfairLock()

    // Memory budget: 20% of Metal recommended working set
    private let maxMemoryBytes: Int
    private var currentMemoryBytes: Int = 0

    // Trie for token-sequence lookup
    private var root = TrieNode()

    // Flat entry store for O(1) access and eviction scanning
    private var entries: [UUID: CacheEntry] = [:]

    // Stats for monitoring
    private var stats = Stats()

    private init() {
        self.maxMemoryBytes = Self.computeMemoryBudget()
    }
}
```

### TrieNode

```swift
private final class TrieNode {
    var children: [Int: TrieNode] = [:]  // token ID → child node
    var entryId: UUID? = nil              // non-nil = a KV cache is stored at this prefix
}
```

### CacheEntry

> **Sendability:** `[KVCache]` contains `MLXArray` which is not `Sendable`. `CacheEntry` is stored inside `TokenPrefixCache` which is `@unchecked Sendable` with all access guarded by `OSAllocatedUnfairLock`. The `[KVCache]` is only accessed under the lock or transferred into `container.perform { }` closures via `nonisolated(unsafe)` — same safety pattern as `ChatSession`'s internal `SerialAccessContainer<Cache>`.
>
> **Checkout model:** Entries are removed from the trie on lookup (see Cache Lease Lifecycle). There are no borrowed/in-flight references — once checked out, the caller has exclusive ownership. This eliminates the need for `inFlightRequests` tracking and avoids shared-mutation hazards.

```swift
private struct CacheEntry {
    let id: UUID
    let modelId: String
    let kvCache: [KVCache]        // the actual cached KV state (not Sendable — guarded by lock)
    let tokenCount: Int            // number of real tokens (excludes image sentinels)
    let cacheKey: [Int]            // extended key: tokens + image fingerprint sentinels
    let estimatedBytes: Int        // memory cost from trimmed KVCache.state[].nbytes
    let createdAt: Date
    var lastAccessAt: Date
    var hitCount: Int
}
```

### Memory Budget

```swift
private static func computeMemoryBudget() -> Int {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return 512 * 1024 * 1024  // 512 MB fallback
    }
    // 20% of Metal recommended working set, matching vllm-mlx
    let budget = Int(Double(device.recommendedMaxWorkingSetSize) * 0.20)
    // Clamp: 256 MB min, 8 GB max
    return max(256 * 1024 * 1024, min(budget, 8 * 1024 * 1024 * 1024))
}
```

### KV Cache Memory Estimation

```swift
/// Estimate the memory footprint of a [KVCache] array.
/// Uses the cache's own state arrays to compute actual size rather than
/// relying on heuristic bytes-per-token estimates.
///
/// **Important:** Call this AFTER trimCacheToOffset() — otherwise you'll
/// measure the pre-allocated buffer size, not the actual used size.
///
/// **vllm-mlx lesson:** Avoid calling .nbytes on lazy MLX arrays as it forces
/// evaluation of the entire computation graph, causing a VRAM spike. Use
/// shape + dtype metadata to compute the same value without triggering eval.
/// After trimCacheToOffset() has called eval(), .nbytes is safe.
private static func estimateBytes(_ kvCache: [KVCache]) -> Int {
    var total = 0
    for layer in kvCache {
        for arr in layer.state {
            // After trim + eval, nbytes reflects actual buffer size.
            // For un-trimmed caches, use shape-based estimation:
            //   math.prod(arr.shape) * arr.dtype.size
            total += arr.nbytes
        }
    }
    return max(total, 1024)  // floor at 1 KB to avoid zero-size entries
}
```

### Core Methods

#### `lookup(cacheKey:modelId:) -> CacheLease`

The `cacheKey` is the extended key from `buildCacheKey()` — it includes both token IDs and image fingerprint sentinels (see Section 3). For text-only requests, it's just the raw token array.

```swift
struct CacheLease: Sendable {
    let entryId: UUID
    let kvCache: [KVCache]?     // nil = cache miss, caller must create fresh
    let matchedTokenCount: Int   // how many real tokens (not sentinels) the cache covers
    let isHit: Bool
}

func lookup(cacheKey: [Int], modelId: String) -> CacheLease {
    lock.lock()
    defer { lock.unlock() }

    let now = Date()
    pruneExpiredLocked(now: now)

    // Walk the trie, tracking the deepest usable match.
    // The trie key includes image fingerprint sentinels (negative values),
    // but matchedTokenCount only counts real tokens (>= 0) because that's
    // what the KV cache offset tracks.
    var node = root
    var bestMatch: (entryId: UUID, depth: Int, realTokenCount: Int)? = nil
    var realTokensSoFar = 0

    for (depth, key) in cacheKey.enumerated() {
        guard let child = node.children[key] else { break }
        node = child

        // Count only real tokens (sentinels are negative)
        if key >= 0 { realTokensSoFar += 1 }

        if let eid = node.entryId,
           let entry = entries[eid],
           entry.modelId == modelId {
            bestMatch = (eid, depth + 1, realTokensSoFar)
        }
    }

    if let match = bestMatch {
        let entry = entries[match.entryId]!

        // CHECKOUT: remove the entry from the trie and the entry store.
        // The caller gets exclusive ownership of the [KVCache] objects.
        // generate() will mutate them in place. After generation, the caller
        // stores the cache at its new (longer) key via store().
        // See Cache Lease Lifecycle.
        removeEntryLocked(entry)

        stats.totalHits += 1

        return CacheLease(
            entryId: match.entryId,
            kvCache: entry.kvCache,
            matchedTokenCount: match.realTokenCount,
            isHit: true
        )
    }

    // Miss — caller will create fresh cache via model.newCache()
    let entryId = UUID()
    stats.totalMisses += 1

    return CacheLease(
        entryId: entryId,
        kvCache: nil,
        matchedTokenCount: 0,
        isHit: false
    )
}
```

> **Note:** This basic trie lookup finds entries stored at nodes along the exact query path (prefix matching). [Section 12](#12-advanced-cache-matching) adds supersequence matching and LCP (Longest Common Prefix) matching to the trie, which handle additional patterns like reusing a longer cache for a shorter request and matching divergent conversations that share a common prefix.

#### `store(entryId:kvCache:cacheKey:modelId:)`

```swift
func store(
    entryId: UUID,
    kvCache: [KVCache],
    cacheKey: [Int],       // extended key (tokens + image fingerprint sentinels)
    modelId: String
) {
    lock.lock()
    defer { lock.unlock() }

    let estimatedBytes = Self.estimateBytes(kvCache)

    // Build/walk trie path using the extended cache key
    var node = root
    for key in cacheKey {
        if node.children[key] == nil {
            node.children[key] = TrieNode()
        }
        node = node.children[key]!
    }

    // If this node already has an entry, remove the old one
    if let oldId = node.entryId, let old = entries[oldId] {
        currentMemoryBytes -= old.estimatedBytes
        entries.removeValue(forKey: oldId)
    }

    node.entryId = entryId

    entries[entryId] = CacheEntry(
        id: entryId,
        modelId: modelId,
        kvCache: kvCache,
        tokenCount: cacheKey.filter({ $0 >= 0 }).count,  // real tokens only
        cacheKey: cacheKey,
        estimatedBytes: estimatedBytes,
        createdAt: Date(),
        lastAccessAt: Date(),
        hitCount: 0
    )

    currentMemoryBytes += estimatedBytes
    enforceBudgetLocked()
}
```

#### No `release()` or `abandon()` needed

With the checkout model, `lookup()` removes the entry from the trie on hit. The caller owns the `[KVCache]` exclusively. After generation:
- **Success:** Call `store()` to insert at the new key.
- **Failure/disconnect:** Simply drop the `[KVCache]` — it will be deallocated. No cache cleanup needed since the entry was already removed from the trie.

This eliminates the `inFlightRequests` bookkeeping and the associated race conditions.

#### `invalidateAll()`

```swift
func invalidateAll() {
    lock.lock()
    defer { lock.unlock() }

    stats.totalEvictions += entries.count
    entries.removeAll()
    root = TrieNode()
    currentMemoryBytes = 0
}
```

#### Eviction

```swift
private let idleTTL: TimeInterval = 30 * 60  // 30 minutes (up from 10)

private func pruneExpiredLocked(now: Date) {
    let expired = entries.values.filter {
        now.timeIntervalSince($0.lastAccessAt) > idleTTL
    }
    for entry in expired {
        removeEntryLocked(entry)
    }
}

private func enforceBudgetLocked() {
    while currentMemoryBytes > maxMemoryBytes {
        // LRU: evict oldest-accessed entry
        guard let victim = entries.values
            .min(by: { $0.lastAccessAt < $1.lastAccessAt })
        else { break }

        removeEntryLocked(victim)
    }
}

private func removeEntryLocked(_ entry: CacheEntry) {
    // Walk trie to remove the entry's node reference
    var node = root
    var path: [(parent: TrieNode, key: Int)] = []
    for key in entry.cacheKey {
        guard let child = node.children[key] else { break }
        path.append((parent: node, key: key))
        node = child
    }
    node.entryId = nil

    // Prune empty trie nodes bottom-up
    for (parent, key) in path.reversed() {
        if let child = parent.children[key],
           child.children.isEmpty && child.entryId == nil {
            parent.children.removeValue(forKey: key)
        } else {
            break
        }
    }

    currentMemoryBytes -= entry.estimatedBytes
    entries.removeValue(forKey: entry.id)
    stats.totalEvictions += 1
}
```

#### Monitoring Snapshot

```swift
struct Snapshot: Sendable {
    let totalEntries: Int
    let totalCachedTokens: Int
    let estimatedBytes: Int
    let memoryBudgetBytes: Int
    let memoryUsagePercent: Double
    let totalHits: Int
    let totalMisses: Int
    let totalEvictions: Int
    let hitRate: Double
}

func snapshot() -> Snapshot {
    lock.lock()
    defer { lock.unlock() }
    pruneExpiredLocked(now: Date())

    let allEntries = Array(entries.values)
    let hits = stats.totalHits
    let misses = stats.totalMisses

    return Snapshot(
        totalEntries: allEntries.count,
        totalCachedTokens: allEntries.reduce(0) { $0 + $1.tokenCount },
        estimatedBytes: currentMemoryBytes,
        memoryBudgetBytes: maxMemoryBytes,
        memoryUsagePercent: maxMemoryBytes > 0 ? Double(currentMemoryBytes) / Double(maxMemoryBytes) * 100 : 0,
        totalHits: hits,
        totalMisses: misses,
        totalEvictions: stats.totalEvictions,
        hitRate: (hits + misses) > 0 ? Double(hits) / Double(hits + misses) * 100 : 0
    )
}
```

### KV Cache Trimming for Partial Reuse

When a trie match covers N tokens but the new prompt has M > N tokens, we need to prefill only the remaining M−N tokens. The framework handles this naturally via `TokenIterator.init()`, which calls `model.prepare(input, cache: cache, windowSize: nil)`. This checks `cache[0].offset` vs `input.text.tokens.count` — if `cache.offset < input.tokens.count`, it only processes the uncached suffix.

**Always pass the full LMInput** and let the framework handle the delta internally:

```swift
// Full prompt tokens: [t0, t1, t2, ..., t_{M-1}]
// Cache covers: [t0, t1, ..., t_{N-1}]  (N = matchedTokenCount)
// Framework will prefill only: [t_N, t_{N+1}, ..., t_{M-1}]

// Pass full LMInput + cached KV — framework skips the cached prefix automatically
let stream = try await container.perform { context in
    try MLXLMCommon.generate(
        input: fullLMInput,     // full prompt, NOT the delta
        cache: cachedKV,        // offset == N, framework prefills only the remaining M-N tokens
        parameters: params,
        context: context
    )
}
```

> **Verified:** `TokenIterator.init(input:model:cache:parameters:)` calls `model.prepare(input, cache: cache, windowSize: nil)`. If `cache[0].offset < input.text.tokens.count`, the model runs prefill only on the uncached suffix. For VL models, this also means the vision encoder is skipped entirely when the cached KV already covers the image tokens. Do NOT construct a delta `LMInput` manually — pass the full input and let the framework handle it.

### KV Cache Trim-to-Offset Before Storage

Following vllm-mlx's `_trim_to_offset()` pattern: KV arrays are often pre-allocated larger than the actual sequence length (e.g., 4096 slots when only 100 are filled). Before storing a `[KVCache]` in `TokenPrefixCache`, trim each layer's K/V tensors to the actual used size:

```swift
import MLX

/// Trim KV cache arrays to their actual used size (offset) before storage.
/// Without this, cached entries waste memory on empty pre-allocated space.
/// Evaluates the sliced arrays so the original large buffers can be freed.
private static func trimCacheToOffset(_ cache: [KVCache]) -> [KVCache] {
    // Check if any layer needs trimming
    let needsTrim = cache.contains { layer in
        let state = layer.state
        guard state.count >= 2 else { return false }
        let seqLen = state[0].dim(2)  // keys shape: [batch, heads, seq, dim]
        return layer.offset > 0 && layer.offset < seqLen
    }
    guard needsTrim else { return cache }

    // Trim each layer's K/V arrays to offset and eval to release original buffers
    var evalTargets: [MLXArray] = []
    for layer in cache {
        let state = layer.state
        guard state.count >= 2 else { continue }
        let offset = layer.offset
        let seqLen = state[0].dim(2)
        if offset > 0 && offset < seqLen {
            // Slice keys and values: [:, :, :offset, :]
            let trimmedKeys = state[0][0..., 0..., 0..<offset, 0...]
            let trimmedValues = state[1][0..., 0..., 0..<offset, 0...]
            layer.state = [trimmedKeys, trimmedValues]
            evalTargets.append(contentsOf: [trimmedKeys, trimmedValues])
        }
    }
    if !evalTargets.isEmpty {
        MLX.eval(evalTargets)
    }
    return cache
}
```

This is called in the store path:

```swift
// After generation completes, before storing in cache:
let trimmedCache = Self.trimCacheToOffset(result.kvCache)
TokenPrefixCache.shared.store(
    entryId: lease.entryId,
    kvCache: trimmedCache,
    cacheKey: fullCacheKey,
    modelId: currentModelId
)
```

---

## 5. Prompt Builder

### New Type: `PromptBuilder`

Create file: `MLXServer/Server/PromptBuilder.swift`

This converts OpenAI API messages into the format that `UserInputProcessor.prepare()` expects, and handles all the message pre-processing that currently lives in `handleChatCompletions()`.

```swift
import MLXLMCommon

/// Converts OpenAI-format API messages into UserInput for mlx-swift-lm.
/// Handles system prompts, tool definitions, message role mapping,
/// and image extraction.
enum PromptBuilder {

    struct PreparedPrompt: Sendable {
        let userInput: UserInput
        let estimatedTextBytes: Int
        let imageCount: Int
    }

    /// Build a UserInput from an API request.
    ///
    /// - Parameters:
    ///   - request: The decoded API request
    ///   - modelId: Current model's repo ID (for tool format selection)
    ///   - thinkingEnabled: Whether thinking mode is on
    /// - Returns: A PreparedPrompt ready for UserInputProcessor.prepare()
    static func build(
        from request: APIChatCompletionRequest,
        modelId: String,
        thinkingEnabled: Bool
    ) -> PreparedPrompt {

        // 1. Collect system messages into instructions
        var instructions = ""
        for msg in request.messages where msg.role == "system" {
            let text = msg.content?.textContent ?? ""
            if !text.isEmpty {
                if !instructions.isEmpty { instructions += "\n\n" }
                instructions += text
            }
        }

        // 2. Build tool prompt (if using manual strategy — see Section 5)
        if let tools = request.tools, !tools.isEmpty {
            let toolPrompt = ToolPromptBuilder.buildSystemPrompt(
                tools: tools, modelId: modelId
            )
            if !instructions.isEmpty { instructions += "\n\n" }
            instructions += toolPrompt
        }

        // 3. Convert non-system messages to Chat.Message format
        let isQwen = modelId.lowercased().contains("qwen")
        var chatMessages: [Chat.Message] = []
        var images: [UserInput.Image] = []
        var estimatedBytes = instructions.utf8.count

        for msg in request.messages where msg.role != "system" {
            let role: Chat.Message.Role = (msg.role == "assistant") ? .assistant : .user

            var text = msg.content?.textContent ?? ""

            // Format tool results for the model
            if msg.role == "tool" {
                if !isQwen {
                    text = "```tool_output\n\(text)\n```"
                }
            }

            // Format assistant tool_calls back into model-native format
            if msg.role == "assistant", let toolCalls = msg.tool_calls, !toolCalls.isEmpty {
                let formatted = isQwen
                    ? ToolPromptBuilder.formatQwenToolCalls(toolCalls)
                    : ToolPromptBuilder.formatGemmaToolCalls(toolCalls)
                text = text.isEmpty ? formatted : text + "\n" + formatted
            }

            // Extract images
            let imageURLs = msg.content?.imageURLs ?? []
            var messageImages: [UserInput.Image] = []
            for urlString in imageURLs {
                if let decoded = ImageDecoder.decode(urlString) {
                    messageImages.append(decoded.image)
                    estimatedBytes += decoded.estimatedBytes
                }
            }

            chatMessages.append(Chat.Message(role: role, content: text, images: messageImages))
            images.append(contentsOf: messageImages)
            estimatedBytes += text.utf8.count
        }

        // 4. Build the system message as a Chat.Message at the front
        var allMessages: [Chat.Message] = []
        if !instructions.isEmpty {
            allMessages.append(Chat.Message(role: .system, content: instructions))
        }
        allMessages.append(contentsOf: chatMessages)

        // 5. Build UserInput
        var additionalContext: [String: any Sendable]? = nil
        if !thinkingEnabled {
            additionalContext = ["enable_thinking": false]
        }

        let userInput = UserInput(
            prompt: .chat(allMessages),
            images: images,
            videos: [],
            tools: nil,  // handled manually via ToolPromptBuilder for now
            additionalContext: additionalContext
        )

        return PreparedPrompt(
            userInput: userInput,
            estimatedTextBytes: estimatedBytes,
            imageCount: images.count
        )
    }
}
```

### Image Decoding Helper

Move the existing `decodeBase64Image()` from `APIServer` to a standalone utility:

```swift
/// Extracted from APIServer — decodes data URIs to UserInput.Image.
enum ImageDecoder {
    struct DecodedImage {
        let image: UserInput.Image
        let estimatedBytes: Int
    }

    static func decode(_ urlString: String) -> DecodedImage? {
        // Same logic as current APIServer.decodeBase64Image()
        // ...
    }
}
```

---

## 6. Native Template Tool Formatting

### Problem

`ToolPromptBuilder` manually injects tool definitions into the system prompt. The model's Jinja chat template may already support tools natively via a `tools` parameter. Double-formatting is fragile.

### Design

`UserInput` has a `.tools: [ToolSpec]?` field. When `mlx-swift-lm`'s `UserInputProcessor.prepare()` receives tools, it passes them to the template's `tools` variable. This is the native path.

#### Strategy Detection

```swift
enum ToolFormattingStrategy {
    /// Pass tools via UserInput.tools — the template handles formatting.
    case templateNative
    /// Inject via ToolPromptBuilder into system prompt text.
    case manualPrompt
}

extension ToolFormattingStrategy {
    /// Detect whether the loaded model's template supports native tool formatting.
    static func detect(for container: ModelContainer) async -> ToolFormattingStrategy {
        // Attempt to prepare a minimal input with a dummy tool.
        // If the processor handles it without error and the output differs
        // from a no-tools run, the template supports tools natively.
        //
        // Alternatively, check the tokenizer_config.json for a "tools"
        // variable in the chat_template string.
        //
        // For now, prefer .templateNative when UserInput.tools is non-nil,
        // since mlx-swift-lm's UserInputProcessor already passes it through.
        // Fall back to .manualPrompt only if we detect breakage.
        return .templateNative
    }
}
```

#### Updated PromptBuilder with Strategy

When using `.templateNative`:

```swift
// In PromptBuilder.build():
if strategy == .templateNative, let apiTools = request.tools, !apiTools.isEmpty {
    // Convert APIToolDefinition → ToolSpec (mlx-swift-lm's type)
    let toolSpecs = apiTools.map { apiTool -> ToolSpec in
        ToolSpec(
            name: apiTool.function.name,
            description: apiTool.function.description ?? "",
            parameters: apiTool.function.parameters  // convert AnyCodable → appropriate type
        )
    }
    userInput.tools = toolSpecs
    // Do NOT append ToolPromptBuilder text to instructions
} else if let apiTools = request.tools, !apiTools.isEmpty {
    // Manual fallback — same as current approach
    let toolPrompt = ToolPromptBuilder.buildSystemPrompt(tools: apiTools, modelId: modelId)
    instructions += "\n\n" + toolPrompt
}
```

#### Output Parsing Hierarchy

For parsing tool calls from model output, use this priority:

1. **Framework `ToolCall` events** — `Generation.toolCall(let call)` from `generate()`. These come from `mlx-swift-lm`'s native tool call parsing. Highest trust.
2. **`ToolCallParser`** — Our manual parser for `<tool_call>` tags, ` ```tool_code``` ` blocks, and bare calls. Used as fallback when the framework doesn't emit `.toolCall` events.

This is already how `APIServer.resolveAssistantResponse()` works (framework calls first, then text parsing). No change needed to the parsing hierarchy.

---

## 7. Client Disconnect Detection

### Problem

When an API client disconnects mid-stream, the server continues generating tokens until completion, wasting GPU cycles.

### Design

#### CancellationToken

```swift
/// Thread-safe cancellation flag. Checked per-token in the generation loop.
final class CancellationToken: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()
    private var _isCancelled = false

    var isCancelled: Bool {
        lock.withLock { _isCancelled }
    }

    func cancel() {
        lock.withLock { _isCancelled = true }
    }
}
```

#### NWConnection State Monitoring

Wire up in `handleStreamingResponse`, before starting generation:

```swift
let cancellation = CancellationToken()

// Monitor for client disconnect
connection.stateUpdateHandler = { state in
    if case .cancelled = state { cancellation.cancel() }
    if case .failed = state { cancellation.cancel() }
}
```

#### Generation Loop Integration

In the streaming loop (inside `InferenceEngine.stream()` or the `runStreamingLoop` equivalent):

```swift
for await event in generationStream {
    if cancellation.isCancelled {
        print("[APIServer] Client disconnected, stopping generation for \(requestId)")
        break  // exit cleanly — do NOT cancel the Task
    }
    continuation.yield(event)
}
```

**Critical:** Do NOT use `task.cancel()` on the underlying generation task. Metal's `mlx::core::eval` can trigger assertion failures if cancelled mid-computation. Instead, just stop consuming the stream and let it drain naturally. The KV cache from the partial generation is still valid and can be stored for reuse.

#### Clean Up After Disconnect

```swift
if cancellation.isCancelled {
    // The working KV cache is still valid — it contains the full prompt
    // plus however many tokens were generated before disconnect.
    // Store it at the prompt key so the prefix is reusable.
    // (The checkout model already removed the old entry from the trie.)
    let trimmedCache = Self.trimCacheToOffset(handle.workingCache)
    TokenPrefixCache.shared.store(
        entryId: lease.entryId,
        kvCache: trimmedCache,
        cacheKey: prepResult.cacheKey,
        modelId: modelId
    )
}
```

---

## 8. Optimized SSE Encoder

### Problem

`JSONEncoder().encode(chunk)` is called per token — allocates encoder, dictionary, and JSON buffer each time.

### Design

#### `StreamingSSEEncoder`

Create file: `MLXServer/Server/StreamingSSEEncoder.swift`

```swift
/// Pre-computes static JSON parts for SSE streaming.
/// Only the dynamic content delta is serialized per token.
struct StreamingSSEEncoder: Sendable {
    private let prefix: Data    // everything before the delta content value
    private let midfix: Data    // between content value and finish_reason
    private let suffix: Data    // closing braces + SSE newlines

    init(requestId: String, created: Int, modelName: String) {
        let escapedModel = Self.escapeJSON(modelName)

        // Pre-build: data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"
        let pre = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(created),\"model\":\"\(escapedModel)\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
        self.prefix = Data(pre.utf8)

        // Between content close and finish_reason
        let mid = "\"},\"finish_reason\":null}]}"
        self.midfix = Data(mid.utf8)

        // SSE event terminator
        self.suffix = Data("\n\n".utf8)
    }

    /// Hot path: encode a single token's text into a complete SSE event.
    /// Zero allocations beyond the output Data.
    func encodeContentDelta(_ text: String) -> Data {
        let escaped = Self.escapeJSON(text)
        var data = Data(capacity: prefix.count + escaped.utf8.count + midfix.count + suffix.count)
        data.append(prefix)
        data.append(contentsOf: escaped.utf8)
        data.append(midfix)
        data.append(suffix)
        return data
    }

    /// Encode the initial role delta (called once per stream).
    func encodeRoleDelta(_ role: String) -> Data {
        let json = "data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"\(role)\"},\"finish_reason\":null}]}"
        return Data((json + "\n\n").utf8)
    }

    /// Final chunk — infrequent, use JSONEncoder for correctness.
    static func encodeFinalChunk(_ chunk: APIChatCompletionChunk) -> Data {
        guard let json = try? JSONEncoder().encode(chunk),
              let str = String(data: json, encoding: .utf8) else {
            return Data("data: {}\n\n".utf8)
        }
        return Data("data: \(str)\n\n".utf8)
    }

    /// Minimal JSON string escaping — handles the characters that appear in model output.
    static func escapeJSON(_ s: String) -> String {
        var result = ""
        result.reserveCapacity(s.count + 8)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"":  result += "\\\""
            case "\\":  result += "\\\\"
            case "\n":  result += "\\n"
            case "\r":  result += "\\r"
            case "\t":  result += "\\t"
            default:
                if scalar.value < 0x20 {
                    result += String(format: "\\u%04x", scalar.value)
                } else {
                    result.append(Character(scalar))
                }
            }
        }
        return result
    }
}
```

#### Integration in Streaming Loop

```swift
let encoder = StreamingSSEEncoder(requestId: requestId, created: created, modelName: modelName)

// Send role delta (once)
await Self.sendData(connection: connection, data: encoder.encodeRoleDelta("assistant"))

// Hot loop — one call per token
for try await generation in stream {
    switch generation {
    case .chunk(let text):
        await Self.sendData(connection: connection, data: encoder.encodeContentDelta(text))
    // ...
    }
}

// Final chunk (once)
let finalData = StreamingSSEEncoder.encodeFinalChunk(finalChunk)
await Self.sendData(connection: connection, data: finalData)
await Self.sendData(connection: connection, data: Data("data: [DONE]\n\n".utf8))
```

---

## 9. Qwen3 EOS Token Fix

### Problem

Qwen3 changed its default `eos_token` from `<|im_end|>` to `<|endoftext|>`, but the chat template still uses `<|im_end|>` as the stop token. Without a fix, Qwen3 may generate past the intended stop point.

### Step 1: Verify

Before implementing, test with a Qwen3 model:
1. Send a simple chat completion request.
2. Check if generation stops at `<|im_end|>` or continues.
3. If it stops correctly, `mlx-swift-lm` already handles this — skip this section.

### Step 2: Fix if Needed

In `ModelManager.swift`, after loading the model:

```swift
// After loading the model container
if config.repoId.lowercased().contains("qwen3") {
    await container.perform { context in
        // Check if the tokenizer's eosTokenId maps to <|endoftext|>
        // If so, override to <|im_end|>'s token ID
        if let imEndId = context.tokenizer.convertTokenToId("<|im_end|>") {
            // Set the model's eosTokenId — exact API depends on mlx-swift-lm version
            // May need to set on the model config or tokenizer
        }
    }
}
```

Alternative: Add `<|im_end|>` as an extra stop token in `GenerateParameters`. Check if `GenerateParameters` supports an `extraEOSTokens` or `additionalEOSTokens` field. If not, this requires an upstream change.

### Affected Files

- `ModelManager.swift`: Post-load EOS override.
- Possibly `InferenceEngine.swift`: Pass extra stop tokens to `GenerateParameters`.

---

## 10. APIServer Rewrite

### Overview

`APIServer.swift` is refactored to use `InferenceEngine` + `TokenPrefixCache` instead of `ChatSession` + `ConversationSessionCache`. The HTTP server shell (NWListener, routing, CORS) stays identical.

### Execution Model: Moving Work Off MainActor

`APIServer` is currently `@MainActor`. The generation loop (`runStreamingLoop`) is already `nonisolated static`, so token output doesn't block SwiftUI. However, the request setup path still runs on MainActor:

| Work | Current | New |
|------|---------|-----|
| JSON request parsing | MainActor | `nonisolated` |
| Base64 image decoding | MainActor (can block UI for large images) | `nonisolated` |
| `PromptBuilder.build()` | MainActor | `nonisolated` (pure function, no actor deps) |
| `InferenceEngine.prepare()` | MainActor (suspends to container queue) | `nonisolated` (suspends to container queue) |
| `TokenPrefixCache.lookup()` | MainActor (fast — lock-protected) | `nonisolated` (fast — lock-protected) |
| `container.perform { generate }` | MainActor (suspends to container queue) | `nonisolated` (suspends to container queue) |
| `requestCount += 1` | MainActor | MainActor (stays — observable state) |
| `LiveCounters` calls | Off-MainActor (OSAllocatedUnfairLock) | Same — no change |

**Implementation:** Extract the entire request processing pipeline into a `nonisolated` method. Only the initial `requestCount += 1` and any observable state updates remain on MainActor:

```swift
@MainActor
private func handleChatCompletions(connection: NWConnection, body: Data?) async {
    requestCount += 1
    // Immediately move off MainActor for all heavy work
    await processCompletionRequest(connection: connection, body: body)
}

/// All request processing runs off MainActor.
/// container.prepare() and container.perform() suspend to the container's
/// internal serial queue. Cache lookup uses OSAllocatedUnfairLock.
/// Image decoding, prompt building, and SSE streaming are pure compute.
nonisolated private func processCompletionRequest(
    connection: NWConnection, body: Data?
) async {
    // ... entire pipeline from JSON parsing through cache store
}
```

This ensures that large base64 payloads, image decoding, and prompt construction never block SwiftUI rendering. The `nonisolated` keyword is sufficient — no `Task.detached` needed, since `container.perform()` already handles its own scheduling.

### Changed Method: `handleChatCompletions()`

The new flow:

```swift
private func handleChatCompletions(connection: NWConnection, body: Data?) async {
    // 1. Parse request (unchanged)
    guard let body, let request = try? JSONDecoder().decode(APIChatCompletionRequest.self, from: body) else { ... }

    // 2. Model swap / reload (unchanged)
    // ...

    // 3. Build prompt via PromptBuilder (handles images per-message for VL models)
    let prepared = PromptBuilder.build(
        from: request,
        modelId: currentModelRepoId,
        thinkingEnabled: Preferences.enableThinking
    )

    // 4. Prepare LMInput (includes .image/.video for VL models) and get tokens
    let engine = InferenceEngine(container: container)
    let prepResult = try await engine.prepare(prepared.userInput)
    // prepResult.lmInput   — full LMInput with text + image pixels
    // prepResult.tokens    — raw token array
    // prepResult.cacheKey  — extended key with image fingerprint sentinels
    // prepResult.hasImages — whether vision processing is needed

    // 5. Look up prefix cache using the extended key (VLM-safe)
    //    For text-only requests, cacheKey == tokens (no overhead).
    //    For VL requests, cacheKey includes image fingerprints so different
    //    images at the same position produce different trie paths.
    let lease = TokenPrefixCache.shared.lookup(
        cacheKey: prepResult.cacheKey,
        modelId: currentModelId
    )

    // 6. Build GenerateParameters
    let params = GenerateParameters(
        maxTokens: maxTokens,
        temperature: Float(temperature),
        topP: Float(topP)
    )

    // 7. Set up cancellation
    let cancellation = CancellationToken()
    if isStream {
        connection.stateUpdateHandler = { state in
            if case .cancelled = state { cancellation.cancel() }
            if case .failed = state { cancellation.cancel() }
        }
    }

    // 8. Run generation via InferenceEngine.stream()
    //    Returns a StreamHandle with both the AsyncStream<Generation> and the
    //    working [KVCache] reference. See Cache Lease Lifecycle.
    //
    //    ALWAYS pass the full LMInput (including .image/.video for VL models).
    //    The framework's model.prepare() checks cache[0].offset vs token count:
    //    - Cache hit: skips prefilled tokens AND skips vision encoder for cached images
    //    - Cache miss: creates fresh cache via model.newCache(), runs full prefill
    let handle = try await engine.stream(
        InferenceEngine.InferenceRequest(
            input: prepResult.lmInput,
            tokens: prepResult.tokens,
            parameters: params,
            cachedKV: lease.kvCache,      // nil for miss, checked-out cache for hit
            cachedTokenCount: lease.matchedTokenCount
        ),
        cancellation: cancellation
    )

    // 9. Stream or collect response (refactored but similar logic)
    if isStream {
        let result = await handleStreamingResponse(
            connection: connection,
            stream: handle.stream,
            requestId: requestId,
            created: created,
            modelName: modelName,
            tools: request.tools,
            cancellation: cancellation,
            isQwen: isQwen
        )
        // ...
    } else {
        let result = await handleNonStreamingResponse(
            connection: connection,
            stream: handle.stream,
            // ...
        )
    }

    // 10. Trim + store cache
    // After generation, handle.workingCache contains K/V states for the full
    // prompt + all generated tokens (mutated in place by generate()).
    // KV arrays may be pre-allocated larger than needed — trim to actual offset
    // before storing so we don't waste cache memory budget on empty space.
    //
    // Store keyed by PROMPT tokens only (prepResult.cacheKey). We do NOT append
    // generated token IDs — the next request's prompt will include the prior
    // assistant response as a message, so prefix matching works naturally.
    // See Cache Lease Lifecycle for rationale.
    if result.succeeded {
        // Trim oversized KV arrays to actual used size (vllm-mlx pattern)
        let trimmedCache = Self.trimCacheToOffset(handle.workingCache)

        TokenPrefixCache.shared.store(
            entryId: lease.entryId,
            kvCache: trimmedCache,
            cacheKey: prepResult.cacheKey,
            modelId: currentModelId
        )
    }
    // On failure: the working cache is dropped. Since lookup() already removed
    // the entry from the trie (checkout model), there's nothing to release.
}
```

### Changed Method: `runStreamingLoop()`

Now takes `cancellation` and the `StreamingSSEEncoder`:

```swift
nonisolated private static func runStreamingLoop(
    connection: NWConnection,
    stream: AsyncStream<Generation>,
    encoder: StreamingSSEEncoder,
    requestId: String,
    cancellation: CancellationToken
) async -> StreamingResult {
    var completionTokens = 0
    var promptTokens = 0
    var fullText = ""
    var frameworkToolCalls: [ToolCall] = []

    do {
        for try await generation in stream {
            if cancellation.isCancelled {
                return StreamingResult(/* ... succeeded: false */)
            }

            switch generation {
            case .chunk(let text):
                completionTokens += 1
                fullText += text
                LiveCounters.shared.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)

                // Hot path: pre-computed SSE
                await sendData(connection: connection, data: encoder.encodeContentDelta(text))

            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                if info.tokensPerSecond > 0 {
                    LiveCounters.shared.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                }

            case .toolCall(let call):
                frameworkToolCalls.append(call)
            }
        }
    } catch {
        // ... error handling
    }

    return StreamingResult(
        promptTokens: promptTokens,
        completionTokens: completionTokens,
        fullText: fullText,
        frameworkToolCalls: frameworkToolCalls,
        succeeded: true
    )
}
```

### Methods Removed from APIServer

- `decodeBase64Image()` → moved to `ImageDecoder` utility
- All `ChatSession` construction logic
- All `ConversationSessionCache` calls
- `messageSignature()` (replaced by token-level cache keying)
- `normalizedAssistantHistoryContent()` (no longer needed — cache is by tokens, not by message content hashes)

### Methods Kept Unchanged

- `start()` / `stop()` — NWListener lifecycle
- `handleConnection()` / `receiveFullHTTPRequest()` / `processHTTPRequest()` — HTTP parsing
- `handleListModels()` — model listing
- `sendResponse()` / `sendData()` / `sendSSEEvent()` — HTTP response helpers (sendSSEEvent only used for non-hot-path chunks)
- `corsHeaders()` — CORS
- `resolveAssistantResponse()` — tool call resolution from framework + text parsing

---

## 11. Statistics & Monitoring Upgrade

The switch to the lower-level API gives us access to data that was previously hidden inside `ChatSession`. This section specifies what new metrics become available and how to surface them.

### What's Currently Tracked

The existing monitoring stack is solid:
- **LiveCounters**: Thread-safe singleton tracking request phases, token counts, tokens/sec, cumulative durations.
- **InferenceStats**: 1Hz polling bridge to SwiftUI with 11 ring-buffer time-series histories.
- **MonitorView**: 6 charts, 2 gauges, 4 cards, cumulative stats, and a per-session list.

### New Metrics Enabled by Direct API Access

| Metric | Source | Current | New |
|--------|--------|---------|-----|
| **Prefill tok/s** | `GenerateCompletionInfo.promptTime` / `promptTokenCount` | Not available — ChatSession didn't expose prefill timing separately | `info.promptTime` gives exact wall-clock prefill duration; compute `promptTokenCount / promptTime` |
| **Time-to-first-token (TTFT)** | Time from request start to first `.chunk` event | Approximated via phase elapsed | Exact: record `Date()` at request start, record `Date()` at first `.chunk` yield, difference = TTFT |
| **Cache prefix match depth** | `CacheLease.matchedTokenCount` | Only hit/miss binary | Exact token count showing how much of the prompt was served from cache vs prefilled fresh |
| **Cache memory budget utilization** | `TokenPrefixCache.snapshot()` | Estimated bytes (heuristic) | Actual bytes from `KVCache.state[].nbytes` — real Metal buffer sizes, not estimates |
| **Prefill tokens saved** | `matchedTokenCount` on cache hit | Approximated via "reuse tokens" | Exact: `matchedTokenCount` = tokens skipped, `totalTokens - matchedTokenCount` = tokens actually prefilled |
| **Vision encoder time** | Timing around `model.prepare()` for VL models | Not available | Time the `container.prepare()` call; for VL models this includes vision encoder. Subtract from total TTFT to isolate language prefill time. |
| **KV cache entries by depth** | Trie structure | Not meaningful (message count) | Token-level depth — shows distribution of how deep cache entries are |
| **Stop reason** | `GenerateCompletionInfo.stopReason` | Only "stop" or "tool_calls" (inferred) | Exact: `.stop` (EOS), `.length` (maxTokens), `.cancelled` (client disconnect) |
| **Per-request cache hit quality** | `matchedTokenCount / totalPromptTokens` | Binary hit/miss | Percentage: "80% of this prompt was cached" — much more informative |
| **Memory pressure events** | `TokenPrefixCache` eviction during `enforceBudget()` | Eviction count only | Eviction count + bytes freed + which entries were evicted (oldest-access-time tracking) |

### LiveCounters Changes

```swift
// ADD to LiveCounters:
private var _prefillTokensPerSecond: Double = 0
private var _timeToFirstToken: TimeInterval = 0
private var _cacheMatchDepth: Int = 0         // tokens matched from cache
private var _totalPrefillTokens: Int = 0       // prompt tokens actually prefilled (not cached)
private var _totalCachedTokens: Int = 0        // prompt tokens served from cache
private var _lastStopReason: String = ""
private var _visionEncoderTime: TimeInterval = 0  // VL models only
private var _totalDisconnects: Int = 0

// ADD to Snapshot:
let prefillTokensPerSecond: Double
let timeToFirstToken: TimeInterval
let cacheMatchDepth: Int
let totalPrefillTokens: Int
let totalCachedTokens: Int
let lastStopReason: String
let visionEncoderTime: TimeInterval
let totalDisconnects: Int

// NEW methods:
func firstTokenGenerated(requestId: String) {
    // Called when the first .chunk event arrives
    // Records TTFT = now - requestStartTime
}

func prefillTimingAvailable(requestId: String, prefillTokPerSec: Double) {
    // Called when .info event arrives with promptTime
}

func cacheMatchRecorded(matchedTokens: Int, totalPromptTokens: Int) {
    // Called after cache lookup
}

func visionProcessingCompleted(requestId: String, duration: TimeInterval) {
    // Called after container.prepare() for VL models
}

func disconnectDetected(requestId: String) {
    // Called when CancellationToken triggers
}
```

### InferenceStats Changes

```swift
// ADD to InferenceStats:
var prefillTokensPerSecond: Double = 0
var timeToFirstToken: TimeInterval = 0
var cacheMatchDepth: Int = 0
var cacheMatchPercent: Double = 0         // matchedTokens / totalPromptTokens * 100
var lastStopReason: String = ""
var visionEncoderTime: TimeInterval = 0
var totalDisconnects: Int = 0

// New memory-aware cache stats (from TokenPrefixCache.snapshot()):
var cacheMemoryBudgetBytes: Int = 0       // max allowed
var cacheMemoryUsedBytes: Int = 0         // currently used (real, not estimated)
var cacheMemoryUsagePercent: Double = 0   // used / budget * 100
var cacheHitRate: Double = 0              // hits / (hits + misses) * 100

// ADD time-series histories:
private(set) var ttftHistory: [DataPoint] = []
private(set) var prefillSpeedHistory: [DataPoint] = []
private(set) var cacheMatchDepthHistory: [DataPoint] = []
private(set) var cacheMemoryHistory: [DataPoint] = []    // replaces estimated cacheFootprintHistory
private(set) var visionTimeHistory: [DataPoint] = []

// REMOVE (no longer applicable):
// - warmCacheEntryCount (no "warm" concept in token cache)
// - generatingCacheEntryCount (checkout model means no "in-flight" concept in cache)
// - cachedSessions: [ConversationSessionCache.SessionSummary]
//   (replaced by TokenPrefixCache entries with different shape)
```

### MonitorView Changes

#### New Charts

**Time-to-First-Token (TTFT)**
```
Line chart showing TTFT per request over time.
Y-axis: milliseconds. Color: cyan.
Shows the user-perceived latency — the most important API quality metric.
```

**Prefill Speed (tok/s)**
```
Line + area chart showing prompt processing speed.
Y-axis: tokens/second. Color: blue.
Separate from generation speed — prefill is memory-bound, generation is compute-bound.
Shows how efficiently the model processes long prompts.
```

**Cache Match Quality**
```
Bar chart showing cache match depth per request.
Stacked bars: green = tokens from cache, red = tokens prefilled fresh.
Much more informative than binary hit/miss — shows partial reuse.
```

**Cache Memory Budget**
```
Replaces current "Cache Footprint (est)" chart.
Dual line: used bytes (orange) vs budget ceiling (dashed gray).
Area fill between used and budget shows available headroom.
Y-axis: MB (real values from KVCache.state[].nbytes, not estimates).
```

**Vision Encoder Time** (only shown when VL model loaded)
```
Bar chart showing vision encoder duration per request.
Y-axis: milliseconds. Color: purple.
Only rendered when visionTimeHistory has non-zero values.
Helps identify if image processing is the latency bottleneck.
```

#### Updated Cards

**Cache Card** — replace "Session Cache" card:

```swift
// Old:
// - Entry count, Warm, Active, Est. Footprint, Cached Tokens, Hit Rate

// New:
VStack(alignment: .leading, spacing: 8) {
    Text("Prefix Cache")               // renamed from "Session Cache"
    Text("\(stats.cacheEntryCount)")    // total entries in trie

    LabeledContent("Memory") {
        Text("\(formatByteCount(stats.cacheMemoryUsedBytes)) / \(formatByteCount(stats.cacheMemoryBudgetBytes))")
    }
    LabeledContent("Usage") {
        Text(String(format: "%.0f%%", stats.cacheMemoryUsagePercent))
    }
    LabeledContent("Hit Rate") {
        Text(String(format: "%.1f%%", stats.cacheHitRate))
    }
    LabeledContent("Avg Match") {
        Text(String(format: "%.0f%%", stats.cacheMatchPercent))
        // Shows average prefix reuse percentage — way more useful than raw hit count
    }

    // Memory pressure indicator
    if stats.cacheMemoryUsagePercent > 80 {
        Label("Memory pressure — evicting old entries", systemImage: "exclamationmark.triangle")
            .font(.caption2)
            .foregroundStyle(.orange)
    }
}
```

#### Updated Cumulative Section

Add tiles:
- **Prefill Saved** — total tokens served from cache (green) — quantifies the real value of caching
- **Disconnects** — total client disconnects detected (shows the value of disconnect detection)
- **Vision Time** — total time spent in vision encoder (purple, only for VL models)

Remove or rename:
- "Reused Prefill" → "Tokens From Cache" (more precise language)
- "Rebuilt Prefill" → "Tokens Prefilled" (clearer)

#### Updated Session List

Replace `ConversationSessionCache.SessionSummary` rows with `TokenPrefixCache.CacheEntrySummary`:

```swift
struct CacheEntrySummary: Identifiable, Sendable {
    let id: UUID
    let modelId: String
    let tokenCount: Int           // real tokens in this cache entry
    let estimatedBytes: Int       // actual KV cache bytes from .state[].nbytes
    let hitCount: Int
    let hasImages: Bool           // whether this entry covers image tokens
    let createdAt: Date
    let lastAccessAt: Date
}

// Row shows:
// - Token count (instead of message count — more meaningful)
// - Actual memory footprint (not estimated)
// - Whether images are involved
// - Hit count and age
// - Context usage bar (same as current)
```

### Integration Points

The statistics data flows through this pipeline:

```
[Generation loop]
    → LiveCounters.shared.firstTokenGenerated()       // on first .chunk
    → LiveCounters.shared.tokenGenerated()             // on each .chunk
    → LiveCounters.shared.prefillTimingAvailable()     // on .info event
    → LiveCounters.shared.requestCompleted()           // on stream end

[Cache operations]
    → LiveCounters.shared.cacheMatchRecorded()         // after TokenPrefixCache.lookup()

[VL processing]
    → LiveCounters.shared.visionProcessingCompleted()  // after container.prepare()

[Disconnect]
    → LiveCounters.shared.disconnectDetected()         // when CancellationToken fires

[1Hz timer]
    → InferenceStats.recordSample()
        → polls LiveCounters.shared.snapshot()
        → polls TokenPrefixCache.shared.snapshot()     // replaces ConversationSessionCache
        → updates all @Observable properties
        → appends to ring buffer histories
```

The existing zero-blocking architecture (LiveCounters → InferenceStats → MonitorView) stays intact. The only change is what data flows through it.

---

## 12. Advanced Cache Matching

The basic trie lookup (Section 4) handles exact prefix matches — the request's token sequence is a direct extension of a cached entry. This section adds two additional matching strategies from vllm-mlx that significantly improve cache hit rates in real-world usage patterns.

### 12.1 Supersequence Matching

#### Problem

A cached entry has tokens `[A, B, C, D, E]` (e.g., a full conversation with an assistant response) and the incoming request has tokens `[A, B, C]` (e.g., the user is re-sending the same conversation but the latest assistant message isn't included because the client is asking for a fresh response). The basic trie walk finds no match — it walks `A → B → C` but the entry is stored at the `E` node.

This matters for:
- Retry patterns — client retries a request after a failure, sending the same prefix without the failed assistant response.
- Edit patterns — client modifies the last message and re-sends, but the prior context is identical.
- Multi-turn tool use — the same prefix is shared across multiple tool-call rounds; the cache from a completed round (with the assistant's tool-call appended) is longer than the next request's prompt.

#### Design

After the prefix walk, if we didn't find an exact match but did walk the full query key without breaking, scan the subtree below the final node for stored entries. The closest (shallowest) stored entry in the subtree is a supersequence of the query — we can trim its KV cache to the query length.

```swift
/// Extended lookup with supersequence matching.
/// Called after the basic trie walk when no exact prefix match was found
/// but the walk consumed ALL tokens in the query key (no break).
///
/// Searches the subtree below the walk's end node for the shallowest
/// stored entry — that entry's token sequence starts with the query's
/// tokens, so its KV cache covers the query as a prefix.
private func findSupersequenceMatchLocked(
    below node: TrieNode,
    queryRealTokenCount: Int,
    modelId: String,
    now: Date
) -> CacheLease? {
    // BFS to find shallowest entry in subtree
    var queue: [(node: TrieNode, depth: Int, realTokens: Int)] = [(node, 0, queryRealTokenCount)]
    var bestEntry: (entryId: UUID, excess: Int)? = nil

    while !queue.isEmpty {
        let (current, depth, _) = queue.removeFirst()
        if let eid = current.entryId,
           let entry = entries[eid],
           entry.modelId == modelId {
            let excess = entry.tokenCount - queryRealTokenCount
            if excess > 0 {
                // Check if the cache supports trimming
                let canTrim = entry.kvCache.allSatisfy { $0.isTrimmable }
                if canTrim {
                    if bestEntry == nil || excess < bestEntry!.excess {
                        bestEntry = (eid, excess)
                    }
                }
            }
        }
        // Don't go too deep — limit subtree scan to avoid O(N) on wide tries
        if depth < 50 {
            for (_, child) in current.children {
                queue.append((child, depth + 1, 0))
            }
        }
    }

    guard let match = bestEntry else { return nil }

    let entry = entries[match.entryId]!

    // Trim the KV cache: reduce offset by the excess tokens.
    // This makes the cache appear as if it only covers the query's tokens.
    let trimmedCache = Self.trimCacheByOffset(entry.kvCache, trimBy: match.excess)

    // CHECKOUT: remove from trie (same as prefix match)
    removeEntryLocked(entry)

    stats.totalHits += 1
    stats.totalSupersequenceHits += 1

    return CacheLease(
        entryId: match.entryId,
        kvCache: trimmedCache,
        matchedTokenCount: queryRealTokenCount,
        isHit: true
    )
}

/// Trim a [KVCache] by reducing offset on each layer.
/// Mutates in place via KVCache.trim() — safe because the entry is checked out
/// (exclusively owned by the caller). The model's next prefill will recompute
/// the trimmed positions, so there's no stale data issue.
private static func trimCacheByOffset(_ cache: [KVCache], trimBy: Int) -> [KVCache] {
    // Use the KVCache protocol's trim() method which handles both
    // KVCacheSimple and RotatingKVCache correctly.
    for layer in cache {
        layer.trim(trimBy)
    }
    return cache
}
```

#### Integration into `lookup()`

```swift
func lookup(cacheKey: [Int], modelId: String) -> CacheLease {
    lock.lock()
    defer { lock.unlock() }

    let now = Date()
    pruneExpiredLocked(now: now)

    // 1. Walk the trie (existing code) ...
    var node = root
    var bestMatch: (...) = nil
    var realTokensSoFar = 0
    var walkedFullKey = true

    for (depth, key) in cacheKey.enumerated() {
        guard let child = node.children[key] else {
            walkedFullKey = false
            break
        }
        node = child
        if key >= 0 { realTokensSoFar += 1 }
        // ... existing bestMatch tracking ...
    }

    // 2. If we have a prefix match, return it (existing code) ...
    if let match = bestMatch { ... }

    // 3. NEW: If we walked the full key without finding a prefix match,
    //    check the subtree for supersequence matches.
    if walkedFullKey {
        if let superLease = findSupersequenceMatchLocked(
            below: node,
            queryRealTokenCount: realTokensSoFar,
            modelId: modelId,
            now: now
        ) {
            return superLease
        }
    }

    // 4. NEW: If no prefix or supersequence match, try LCP (Section 12.2) ...

    // 5. Miss
    let entryId = UUID()
    stats.totalMisses += 1
    return CacheLease(entryId: entryId, kvCache: nil, matchedTokenCount: 0, isHit: false)
}
```

### 12.2 Longest Common Prefix (LCP) Matching

#### Problem

Cached entry has tokens `[SYS, A, B, C, X, Y]` and the incoming request has `[SYS, A, B, C, D, E]`. They share a 4-token prefix `[SYS, A, B, C]` but diverge after that. The basic trie walk reaches `C` and finds no child for `D`, so it stops. If no entry is stored at the `C` node, this is a cache miss — even though we could reuse the KV cache for the first 4 tokens and only prefill `[D, E]`.

This is the **dominant pattern in agentic use**:
- Same system prompt + tool definitions + conversation history, but different final user message.
- Same RAG context prefix, different query.
- Same multi-turn conversation up to the last exchange, different follow-up.

Without LCP matching, every divergent final message causes a full prefill from scratch. With it, only the divergent suffix needs prefilling — often saving thousands of tokens of prefill work.

#### Design

When the trie walk breaks at depth `D` (divergence point), search for the nearest stored entry in the subtree at depth `D−1` (the last matched node). The entry with the most tokens beyond `D−1` gives the deepest common prefix. Trim its KV cache to depth `D` and return the remainder.

```swift
/// LCP matching: find the best entry that shares the longest common prefix
/// with the query but diverges at some point.
///
/// Called when the trie walk broke at position `divergenceDepth` (no child
/// for the next token in the query). We look for stored entries along
/// the path we DID walk — any entry stored between the root and the
/// divergence point has a KV cache that covers a prefix of the query.
///
/// Additionally, we check sibling subtrees at the divergence point:
/// entries stored one branch over may share the same prefix up to the
/// divergence point. Their KV cache can be trimmed to the common prefix.
private func findLCPMatchLocked(
    pathNodes: [(node: TrieNode, key: Int)],
    divergenceDepth: Int,
    realTokensAtDivergence: Int,
    modelId: String,
    now: Date
) -> CacheLease? {
    // Strategy 1: Check sibling entries at the divergence point.
    // The parent node is pathNodes[divergenceDepth - 1].
    // Its children (other than the query's next token) lead to cached entries
    // that share the same prefix up to divergenceDepth.
    guard divergenceDepth > 0, divergenceDepth <= pathNodes.count else { return nil }

    let parentNode = divergenceDepth >= 2
        ? pathNodes[divergenceDepth - 2].node.children[pathNodes[divergenceDepth - 1].key]!
        : pathNodes[0].node

    // Search all children of the parent (including deeper subtrees)
    // for the shallowest stored entry — it shares the most of our prefix.
    var bestEntry: (entryId: UUID, entryTokenCount: Int)? = nil

    for (childKey, childNode) in parentNode.children {
        // Skip the branch we already walked (that's where we diverged)
        if divergenceDepth < pathNodes.count && childKey == pathNodes[divergenceDepth].key {
            continue
        }
        // BFS this sibling subtree for stored entries
        var queue: [TrieNode] = [childNode]
        while !queue.isEmpty {
            let current = queue.removeFirst()
            if let eid = current.entryId,
               let entry = entries[eid],
               entry.modelId == modelId,
               entry.kvCache.allSatisfy({ $0.isTrimmable }) {
                if bestEntry == nil || entry.tokenCount < bestEntry!.entryTokenCount {
                    bestEntry = (eid, entry.tokenCount)
                }
            }
            for (_, child) in current.children {
                queue.append(child)
            }
        }
    }

    guard let match = bestEntry else { return nil }

    let entry = entries[match.entryId]!

    // Trim the cache to the common prefix length.
    // The entry has `entryTokenCount` tokens, but only `realTokensAtDivergence`
    // are shared with the query. Trim the excess.
    let excess = entry.tokenCount - realTokensAtDivergence
    guard excess > 0 else { return nil }

    let trimmedCache = Self.trimCacheByOffset(entry.kvCache, trimBy: excess)

    // CHECKOUT: remove from trie (same as prefix match)
    removeEntryLocked(entry)

    stats.totalHits += 1
    stats.totalLCPHits += 1

    return CacheLease(
        entryId: match.entryId,
        kvCache: trimmedCache,
        matchedTokenCount: realTokensAtDivergence,
        isHit: true
    )
}
```

#### Integration into `lookup()`

The `lookup()` method tracks the trie path as it walks, and passes it to `findLCPMatchLocked` when the walk breaks:

```swift
func lookup(cacheKey: [Int], modelId: String) -> CacheLease {
    lock.lock()
    defer { lock.unlock() }

    let now = Date()
    pruneExpiredLocked(now: now)

    var node = root
    var bestMatch: (entryId: UUID, depth: Int, realTokenCount: Int)? = nil
    var realTokensSoFar = 0
    var pathNodes: [(node: TrieNode, key: Int)] = []
    var divergenceDepth: Int? = nil

    for (depth, key) in cacheKey.enumerated() {
        guard let child = node.children[key] else {
            divergenceDepth = depth
            break
        }
        pathNodes.append((node: node, key: key))
        node = child
        if key >= 0 { realTokensSoFar += 1 }

        if let eid = node.entryId,
           let entry = entries[eid],
           entry.modelId == modelId {
            bestMatch = (eid, depth + 1, realTokensSoFar)
        }
    }

    // 1. Prefix match — checkout (remove) the entry from the trie
    if let match = bestMatch {
        // ... return prefix match lease (same checkout logic as Section 4) ...
    }

    // 2. Supersequence match (Section 12.1)
    if divergenceDepth == nil {
        if let superLease = findSupersequenceMatchLocked(
            below: node,
            queryRealTokenCount: realTokensSoFar,
            modelId: modelId,
            now: now
        ) {
            return superLease
        }
    }

    // 3. LCP match (this section)
    if let depth = divergenceDepth, depth > 0 {
        if let lcpLease = findLCPMatchLocked(
            pathNodes: pathNodes,
            divergenceDepth: depth,
            realTokensAtDivergence: realTokensSoFar,
            modelId: modelId,
            now: now
        ) {
            return lcpLease
        }
    }

    // 4. Miss
    let entryId = UUID()
    stats.totalMisses += 1
    return CacheLease(entryId: entryId, kvCache: nil, matchedTokenCount: 0, isHit: false)
}
```

#### Match Priority

The lookup returns the first successful match in this order:

| Priority | Match Type | When It Fires | What It Saves |
|----------|-----------|---------------|---------------|
| 1 | **Prefix** | Cached `[A,B,C]`, query `[A,B,C,D,E]` | All cached prefix tokens |
| 2 | **Supersequence** | Cached `[A,B,C,D,E]`, query `[A,B,C]` | All query tokens (trim cache) |
| 3 | **LCP** | Cached `[A,B,C,X,Y]`, query `[A,B,C,D,E]` | Shared prefix tokens (trim cache) |

This matches vllm-mlx's `fetch()` priority order: exact → prefix → supersequence → LCP → miss.

### 12.3 Statistics for Advanced Matching

Add to `Stats`:

```swift
private struct Stats {
    var totalHits: Int = 0
    var totalMisses: Int = 0
    var totalEvictions: Int = 0
    var totalPrefixHits: Int = 0           // basic prefix match
    var totalSupersequenceHits: Int = 0    // reused longer cache for shorter request
    var totalLCPHits: Int = 0              // divergent sequences with shared prefix
}
```

Add to `Snapshot`:

```swift
struct Snapshot: Sendable {
    // ... existing fields ...
    let prefixHits: Int
    let supersequenceHits: Int
    let lcpHits: Int
}
```

These are surfaced in MonitorView's cache card:

```swift
LabeledContent("Hit Breakdown") {
    Text("P:\(stats.prefixHits) S:\(stats.supersequenceHits) L:\(stats.lcpHits)")
        .font(.caption)
}
```

### 12.4 Affected Files

| File | Changes |
|------|---------|
| `MLXServer/Server/TokenPrefixCache.swift` | Add `findSupersequenceMatchLocked()`, `findLCPMatchLocked()`, `trimCacheByOffset()`. Extend `lookup()` with new match stages. Add stats fields. |
| `MLXServer/Models/InferenceStats.swift` | Add match-type breakdown fields to snapshot polling. |
| `MLXServer/Views/MonitorView.swift` | Show hit breakdown in cache card. |

---

## 13. KV Cache Quantization

### Problem

Long conversations produce large KV caches. A 4K-token conversation with a 4-bit 4B model generates roughly 200–400 MB of KV state (FP16 keys + values across all layers). With 8 cached conversations, this exceeds the memory budget and triggers eviction — reducing cache effectiveness.

vllm-mlx addresses this with optional KV cache quantization: before storing, convert FP16 K/V tensors to 8-bit (or lower) quantized representations, reducing memory footprint by ~50%. Dequantize on retrieval before passing to the model.

### Design

#### Configuration

```swift
extension TokenPrefixCache {
    struct QuantizationConfig {
        /// Whether to quantize cached KV states. Default: false.
        let enabled: Bool
        /// Quantization bit width. 8-bit is the sweet spot (50% savings, minimal quality loss).
        let bits: Int
        /// Group size for quantization. Matches mlx-swift-lm's default.
        let groupSize: Int
        /// Minimum token count before quantization applies.
        /// Short sequences aren't worth quantizing (overhead > savings).
        let minTokens: Int

        static let `default` = QuantizationConfig(
            enabled: false,
            bits: 8,
            groupSize: 64,
            minTokens: 256
        )
    }
}
```

#### Quantize Before Storage

```swift
import MLX

/// Quantize a [KVCache] for compact storage.
/// Each layer's K/V arrays are quantized from FP16 to `bits`-bit representation.
/// Returns a new array of QuantizedKVCache layers.
///
/// Skips layers that are already quantized or that have non-standard structure
/// (e.g., hybrid Mamba+Transformer models with cumulative RNN state).
private static func quantizeCache(
    _ cache: [KVCache],
    bits: Int,
    groupSize: Int
) -> [KVCache] {
    cache.map { layer in
        // Only quantize standard KVCacheSimple layers with K/V arrays
        guard layer.isTrimmable,
              layer.state.count >= 2 else {
            return layer  // pass through non-standard layers unchanged
        }

        // Use mlx-swift-lm's built-in quantization if available.
        // KVCacheSimple may have a toQuantized() method.
        // If not, quantize manually:
        let keys = layer.state[0]
        let values = layer.state[1]

        let (quantizedKeys, scalesK, biasesK) = MLX.quantized(keys, groupSize: groupSize, bits: bits)
        let (quantizedValues, scalesV, biasesV) = MLX.quantized(values, groupSize: groupSize, bits: bits)

        // Store as a QuantizedKVCache or equivalent wrapper
        // that holds (data, scales, biases) tuples instead of raw arrays
        let qLayer = QuantizedKVCacheWrapper(
            keys: (quantizedKeys, scalesK, biasesK),
            values: (quantizedValues, scalesV, biasesV),
            offset: layer.offset,
            bits: bits,
            groupSize: groupSize
        )
        // Eval the quantized tensors so the originals can be freed
        MLX.eval([quantizedKeys, scalesK, biasesK, quantizedValues, scalesV, biasesV])

        return qLayer
    }
}
```

#### Dequantize on Retrieval

```swift
/// Dequantize a [KVCache] back to FP16 before passing to the model.
/// The model requires standard FP16 K/V tensors for attention computation.
private static func dequantizeCache(_ cache: [KVCache]) -> [KVCache] {
    cache.map { layer in
        guard let qLayer = layer as? QuantizedKVCacheWrapper else {
            return layer  // already standard, pass through
        }

        let keys = MLX.dequantized(
            qLayer.keys.0, scales: qLayer.keys.1, biases: qLayer.keys.2,
            groupSize: qLayer.groupSize, bits: qLayer.bits
        )
        let values = MLX.dequantized(
            qLayer.values.0, scales: qLayer.values.1, biases: qLayer.values.2,
            groupSize: qLayer.groupSize, bits: qLayer.bits
        )

        // Create a standard KVCache with the dequantized tensors
        // The model's attention will use these as normal FP16 K/V
        let standardLayer = KVCacheSimple()
        standardLayer.state = [keys, values]
        standardLayer.offset = qLayer.offset

        MLX.eval([keys, values])

        return standardLayer
    }
}
```

#### Integration into TokenPrefixCache

```swift
// In store():
func store(entryId: UUID, kvCache: [KVCache], cacheKey: [Int], modelId: String) {
    lock.lock()
    defer { lock.unlock() }

    // Trim oversized arrays first (Section 4)
    var cache = Self.trimCacheToOffset(kvCache)

    // Quantize if enabled and sequence is long enough
    if quantizationConfig.enabled,
       cacheKey.filter({ $0 >= 0 }).count >= quantizationConfig.minTokens {
        cache = Self.quantizeCache(cache, bits: quantizationConfig.bits, groupSize: quantizationConfig.groupSize)
    }

    let estimatedBytes = Self.estimateBytes(cache)
    // ... rest of existing store logic ...
}

// In lookup(), when returning a cache hit:
if let match = bestMatch {
    var entry = entries[match.entryId]!
    // ... existing bookkeeping ...

    // Dequantize before returning to caller
    let returnCache = quantizationConfig.enabled
        ? Self.dequantizeCache(entry.kvCache)
        : entry.kvCache

    return CacheLease(
        entryId: match.entryId,
        kvCache: returnCache,
        matchedTokenCount: match.realTokenCount,
        isHit: true
    )
}
```

### Memory Savings

| Scenario | FP16 KV Size | 8-bit Quantized | Savings |
|----------|-------------|-----------------|---------|
| 1K tokens, 32-layer model | ~50 MB | ~25 MB | 50% |
| 4K tokens, 32-layer model | ~200 MB | ~100 MB | 50% |
| 8 cached conversations @ 4K | ~1.6 GB | ~800 MB | 800 MB freed |

The 50% savings means the same memory budget can hold roughly twice as many cached conversations, doubling effective cache hit rates for multi-conversation workloads.

### Quality Impact

8-bit quantization of KV cache has minimal impact on generation quality:
- K/V tensors are attention states, not model weights — they're less sensitive to quantization.
- vllm-mlx uses this in production with no reported quality degradation.
- The `minTokens` threshold (256) avoids quantizing short sequences where the overhead isn't worth it.

### QuantizedKVCacheWrapper

```swift
/// Wrapper that stores quantized KV tensors and conforms to KVCache protocol.
/// Used internally by TokenPrefixCache for compact storage.
final class QuantizedKVCacheWrapper: KVCache {
    let keys: (MLXArray, MLXArray, MLXArray)     // (data, scales, biases)
    let values: (MLXArray, MLXArray, MLXArray)   // (data, scales, biases)
    var offset: Int
    let bits: Int
    let groupSize: Int

    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }  // must dequantize before trimming

    var state: [MLXArray] {
        get { [keys.0, keys.1, keys.2, values.0, values.1, values.2] }
        set { /* read-only for quantized cache */ }
    }

    var metaState: [String] {
        get { ["\(offset)", "\(bits)", "\(groupSize)"] }
        set { }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("QuantizedKVCacheWrapper is read-only — dequantize first")
    }

    func trim(_ n: Int) -> Int {
        // Cannot trim quantized cache — caller must dequantize first
        return 0
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    init(keys: (MLXArray, MLXArray, MLXArray), values: (MLXArray, MLXArray, MLXArray),
         offset: Int, bits: Int, groupSize: Int) {
        self.keys = keys
        self.values = values
        self.offset = offset
        self.bits = bits
        self.groupSize = groupSize
    }
}
```

> **Implementation note:** `mlx-swift-lm` already provides a `QuantizedKVCache` class (in `KVCache.swift`) that conforms to `KVCache` with built-in `quantized()`/`dequantized()` support and proper `trim()` handling. **Use the framework's type instead of this custom wrapper.** It handles edge cases around different model architectures (including hybrid Mamba+Transformer). The `QuantizedKVCacheWrapper` above is shown for illustration; the actual implementation should use `QuantizedKVCache` from `MLXLMCommon`.

> **Interaction with advanced matching (Phase 5):** Quantized entries stored via the framework's `QuantizedKVCache` support `trim()` natively (unlike the custom wrapper above which returned `isTrimmable == false`). Verify that `QuantizedKVCache.trim()` works correctly before enabling both features. If trim is not supported on quantized entries, supersequence and LCP matching will skip them (the `isTrimmable` guard handles this gracefully) — quantized entries still get basic prefix matches. A future enhancement could add a "dequantize → trim → re-quantize" path for advanced matching on quantized entries, but this is not required for initial implementation.

### Affected Files

| File | Changes |
|------|---------|
| `MLXServer/Server/TokenPrefixCache.swift` | Add `QuantizationConfig`, `quantizeCache()`, `dequantizeCache()`, `QuantizedKVCacheWrapper`. Integrate into `store()` and `lookup()`. |
| `MLXServer/Utilities/Preferences.swift` | Add `kvQuantizationEnabled`, `kvQuantizationBits` preferences (default: off, 8). |
| `MLXServer/Views/MonitorView.swift` | Show quantization status in cache card when enabled. |

---

## 14. File-by-File Change Map

### New Files

| File | Purpose |
|------|---------|
| `MLXServer/Server/InferenceEngine.swift` | Stateless inference wrapper around ModelContainer.generate() |
| `MLXServer/Server/TokenPrefixCache.swift` | Trie-based token-level KV cache with memory-aware eviction |
| `MLXServer/Server/PromptBuilder.swift` | OpenAI API messages → UserInput conversion |
| `MLXServer/Server/StreamingSSEEncoder.swift` | Pre-computed JSON encoder for SSE streaming |
| `MLXServer/Server/CancellationToken.swift` | Thread-safe cancellation flag for disconnect detection |
| `MLXServer/Server/ImageDecoder.swift` | Base64 data URI → UserInput.Image (extracted from APIServer) |
| `MLXServer/Server/ImageFingerprint.swift` | Fast image hashing for VL model cache keys (base64 hash or sparse pixel sampling) |

### Modified Files

| File | Changes |
|------|---------|
| `MLXServer/Server/APIServer.swift` | Major refactor: drop ChatSession, use InferenceEngine + TokenPrefixCache + PromptBuilder + StreamingSSEEncoder. Add CancellationToken wiring. Remove ConversationSessionCache calls, messageSignature(), decodeBase64Image(). Add TTFT, prefill speed, cache match, vision timing, and disconnect reporting to LiveCounters. Uses `container.perform { ctx in MLXLMCommon.generate(input:cache:parameters:context:) }` for cache-aware generation. |
| `MLXServer/Server/TokenPrefixCache.swift` | (Phase 5) Add supersequence matching, LCP matching, `trimCacheByOffset()`, match-type stats. (Phase 6) Add `QuantizationConfig`, `QuantizedKVCacheWrapper`, `quantizeCache()`, `dequantizeCache()`. |
| `MLXServer/Models/InferenceStats.swift` | Add new metrics to LiveCounters (TTFT, prefill tok/s, cache match depth, vision time, disconnects). Add new snapshot fields. Add new time-series histories to InferenceStats. Replace ConversationSessionCache.snapshot() polling with TokenPrefixCache.snapshot(). Remove obsolete fields (warmCacheEntryCount, cachedSessions). Add match-type breakdown (prefix/supersequence/LCP). |
| `MLXServer/Views/MonitorView.swift` | Add 3-5 new charts (TTFT, prefill speed, cache match quality, cache memory budget, vision encoder time). Update cache card with memory budget, match %, and hit breakdown. Update cumulative tiles. Replace session list with TokenPrefixCache entry list. Show quantization status when enabled. |
| `MLXServer/ViewModels/ModelManager.swift` | Add Qwen3 EOS token fix (if needed). Expose `modelContainer` for direct API access. |
| `MLXServer/Utilities/Preferences.swift` | Add `kvQuantizationEnabled`, `kvQuantizationBits` preferences (default: off, 8). |
| `MLXServer/Models/ModelConfig.swift` | No structural changes. Optionally add per-model metadata for cache byte estimation. |

### Deleted Files

| File | Reason |
|------|--------|
| `MLXServer/Server/ConversationSessionCache.swift` | Fully replaced by `TokenPrefixCache`. |

### Unchanged Files

| File | Reason |
|------|--------|
| `MLXServer/ViewModels/ChatViewModel.swift` | UI chat path keeps using ChatSession — unrelated to API path. |
| `MLXServer/Server/APIModels.swift` | Codable request/response structs are the same. |
| `MLXServer/Server/ToolPromptBuilder.swift` | Kept as fallback for manual tool prompt injection. |
| `MLXServer/Server/ToolCallParser.swift` | Kept as fallback for text-based tool call parsing. |
| `MLXServer/Views/ChatMessagesView.swift` | Chat UI untouched. |
| `MLXServer/Views/ChatInputView.swift` | Chat input untouched. |
| `MLXServer/Views/DownloadModalView.swift` | Download UI untouched. |
| `MLXServer/Views/StatusBarView.swift` | Status bar untouched (already reads from InferenceStats). |
| `MLXServer/Utilities/*` | No changes needed. |
| `MLXServer/Commands/*` | Menu commands untouched. |

---

## 15. Implementation Order

Each step should be independently buildable and testable.

### Phase 1: Foundation (no behavior change yet)

1. [x] **`CancellationToken.swift`** — Standalone utility, no dependencies. Write + unit test.
2. [x] **`ImageDecoder.swift`** — Extract from APIServer. Mechanical move.
3. [x] **`StreamingSSEEncoder.swift`** — Standalone, testable in isolation. Verify JSON output matches current `JSONEncoder` output.

### Phase 2: Core Engine

4. [x] **`PromptBuilder.swift`** — Convert API messages to UserInput. Test by comparing tokenized output to what ChatSession produces for the same messages.
5. [x] **`TokenPrefixCache.swift`** — The big one. Build trie + eviction + monitoring. Test: insert entries, verify lookup, verify eviction under memory pressure, verify trie cleanup.
6. [x] **`InferenceEngine.swift`** — Thin wrapper using `container.perform { ctx in MLXLMCommon.generate(input:cache:parameters:context:) }`. Test: run a simple prompt through it, verify output matches ChatSession output.

Validation note: `PromptBuilder.swift` is now covered by both shaping-parity unit tests and a model-backed tokenization parity test against the cached local Gemma 3 4B VLM. `InferenceEngine.swift` is now covered by a model-backed smoke test that compares one-token output and prompt-token counts against `ChatSession` on the same locally cached Gemma model.

### Phase 3: Integration

7. [x] **`APIServer.swift` rewrite** — Wire everything together. Replace ChatSession with InferenceEngine, ConversationSessionCache with TokenPrefixCache, add PromptBuilder and StreamingSSEEncoder.
8. [x] **Delete `ConversationSessionCache.swift`** — Only after APIServer is fully migrated and tested.

Validation note: `APIServer.swift` now routes the API path through `PromptBuilder`, `InferenceEngine`, `TokenPrefixCache`, and `StreamingSSEEncoder`, and the full repository test workflow is green. Image-bearing requests now participate in prefix-cache reuse via image-aware cache keys built from prompt tokens plus stable image fingerprints, preventing false hits across different images while enabling same-image reuse.

### Phase 4: Statistics & Monitoring

9. [x] **LiveCounters upgrade** — Add TTFT, prefill tok/s, cache match depth, vision time, disconnect tracking. Wire up new reporting calls in APIServer.
10. [x] **InferenceStats upgrade** — Add new snapshot fields, new time-series histories. Switch from ConversationSessionCache.snapshot() to TokenPrefixCache.snapshot().
11. [x] **MonitorView upgrade** — Add TTFT chart, prefill speed chart, cache match quality chart, cache memory budget chart. Update cache card and cumulative tiles. Add vision encoder time chart (conditional on VL model). Replace session list with cache entry list.

Validation note: `InferenceStats.swift` now samples `TokenPrefixCache` directly and `MonitorView.swift` now surfaces TTFT, prefill speed, cache match depth, cache memory pressure, disconnect totals, vision prepare time, and the prefix/supersequence/LCP hit breakdown from `LiveCounters` and `TokenPrefixCache`.

### Phase 5: Advanced Cache Matching

12. [x] **Supersequence matching** — `TokenPrefixCache` now includes `findSupersequenceMatchLocked()` and `trimCacheByOffset()`, and `lookup()` performs a subtree scan after a full-key walk with no direct entry. Coverage includes both logical cache tests and a model-backed test that verifies the leased KV cache is trimmed to the shorter prefix length.
13. [x] **LCP matching** — `TokenPrefixCache` now includes `findLCPMatchLocked()`, and `lookup()` attempts LCP reuse only on actual divergence. Coverage includes direct cache tests for divergent suffix reuse and shallow-prefix rejection, plus model-backed same-system/different-user reuse validation.
14. [x] **Match stats** — `TokenPrefixCache`, `InferenceStats`, and `MonitorView` now track and surface `prefixHits`, `supersequenceHits`, and `lcpHits` in the cache snapshot and monitor cache card.

### Phase 6: KV Cache Quantization

15. **`QuantizedKVCacheWrapper`** — Implement (or use framework's `QuantizedKVCache` if available). Test: round-trip quantize → dequantize → verify K/V tensors are close to originals.
16. **Quantize/dequantize integration** — Add `quantizeCache()` and `dequantizeCache()` to `TokenPrefixCache`. Wire into `store()` and `lookup()`. Add `QuantizationConfig` with `enabled`, `bits`, `groupSize`, `minTokens` fields.
17. **Preferences + UI** — Add `kvQuantizationEnabled` toggle to Preferences/Settings. Show quantization status in MonitorView cache card.

### Phase 7: Polish

18. **Qwen3 EOS fix** — Verify first, implement if needed.
19. **Native template tool formatting** — Switch from `.manualPrompt` to `.templateNative` once verified working.

---

## 16. Testing Checklist

### Cache Correctness

- [x] Cold start: no cache entries → fresh generation works
- [x] Second identical request → full cache hit, zero prefill tokens
- [x] Conversation continuation (add 1 message) → partial cache hit
- [x] Conversation continuation (add 2+ messages, e.g. tool-use flow) → partial cache hit (not a miss!)
- [x] Same system prompt, different user message → system prompt prefix cached and reused
- [x] Different system prompt → no false cache hit
- [x] Model swap → cache invalidated, fresh generation works
- [x] Idle unload + reload → cache invalidated, fresh generation works

### Memory Management

- [x] Memory budget computed correctly from Metal device
- [x] Entries evicted under memory pressure (oldest first)
- [x] Expired entries pruned after 30 min idle
- [x] Trie nodes cleaned up when entries are evicted (no memory leak)
- [x] `snapshot()` reports accurate memory usage and hit rates

### Disconnect Handling

- [x] Client disconnects mid-stream → generation stops within ~200ms
- [x] Partial KV cache from disconnected request is still stored for reuse
- [x] No Metal assertion failures on disconnect

### Streaming

- [x] SSE JSON is valid and parseable by standard clients
- [x] `StreamingSSEEncoder` output matches `JSONEncoder` output byte-for-byte (for content deltas)
- [x] Role delta sent once at stream start
- [x] Tool call chunks sent correctly
- [x] Final chunk has finish_reason and usage stats
- [x] `data: [DONE]` sent at end

### Tool Use

- [x] Gemma tool_code blocks parsed correctly
- [x] Qwen `<tool_call>` tags parsed correctly
- [x] Framework `ToolCall` events handled correctly
- [x] Tool results round-trip correctly (user sends tool result → model sees it in context)
- [x] finish_reason is "tool_calls" when tools are invoked

### Vision-Language Models

- [ ] Single image + text prompt → correct vision processing → coherent image description
- [ ] Multiple images in a single message → all images processed correctly
- [ ] Image + text in same message → both contribute to response
- [ ] Images in earlier messages, text-only follow-up → cache hit (vision encoder skipped)
- [x] Same conversation, same images → cache hit on subsequent requests
- [x] Same conversation, different image swapped → cache miss, fresh vision processing
- [ ] Text-only conversation on a VL model → no vision overhead, normal cache behavior
- [ ] Large images (4K+) → properly resized by UserInputProcessor, no OOM
- [ ] Base64 data-URI images decoded correctly (PNG, JPEG)
- [x] Image fingerprinting: same image bytes → same fingerprint → cache hit
- [x] Image fingerprinting: different images → different fingerprints → cache miss
- [ ] Non-vision model rejects image inputs with clear error message
- [ ] Mixed: image in user msg 1, assistant response, text-only user msg 2 → cache covers all of msg 1 + response

### Advanced Cache Matching (Section 12)

- [x] Supersequence: cached `[A,B,C,D,E]`, query `[A,B,C]` → cache hit, KV trimmed to 3 tokens
- [ ] Supersequence: cached entry has non-trimmable layers (hybrid model) → graceful skip, falls through to miss
- [ ] Supersequence: multiple candidates in subtree → shallowest (least excess) is chosen
- [x] LCP: cached `[SYS,A,B,X,Y]`, query `[SYS,A,B,D,E]` → cache hit covering `[SYS,A,B]`, remaining `[D,E]`
- [ ] LCP: divergence at depth 0 (no shared prefix at all) → no LCP match, clean miss
- [ ] LCP: multiple sibling entries at divergence → best (shallowest) is chosen
- [ ] LCP agentic pattern: same system prompt (500 tokens) + different user message → system prompt cached and reused
- [x] Match priority: prefix match takes priority over supersequence and LCP
- [ ] Match priority: supersequence takes priority over LCP
- [x] Stats: prefix, supersequence, and LCP hits counted separately in snapshot
- [ ] Trim correctness: KVCache.trim() called with correct excess count, offset reduced accordingly
- [ ] Trim + generate: trimmed cache produces valid generation (no garbled output from stale K/V)

### KV Cache Quantization (Section 13)

- [ ] Round-trip: quantize(8-bit) → dequantize → K/V tensors close to originals (max error < 1%)
- [ ] Memory: quantized entry uses ~50% of FP16 memory (check estimateBytes before/after)
- [ ] Short sequences: entries below `minTokens` threshold are NOT quantized
- [ ] Disabled by default: `QuantizationConfig.default.enabled == false`
- [ ] Store path: quantization happens after trim-to-offset, before memory estimation
- [ ] Lookup path: dequantization happens before returning cache to caller
- [ ] Non-standard layers: hybrid model layers (non-trimmable) passed through unquantized
- [ ] Generation quality: quantized-then-dequantized cache produces coherent output (manual check)
- [ ] Supersequence + quantized: must dequantize before trimming (QuantizedKVCacheWrapper.isTrimmable == false)
- [ ] Preferences: toggle works, changes take effect on next store (existing entries not re-quantized)

### Thinking Mode

- [x] `enable_thinking: false` passed through to template correctly
- [ ] Thinking mode on: `<think>` blocks appear in output
- [ ] Thinking mode off: no `<think>` blocks

### Compatibility

- [x] `GET /health` → `{"status":"ok"}`
- [x] `GET /v1/models` → model list with context windows
- [x] Non-streaming `POST /v1/chat/completions` → full response
- [x] Streaming `POST /v1/chat/completions` → SSE stream
- [x] Model field in request triggers model swap
- [x] UI chat (ChatViewModel) completely unaffected
