import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXVLM
import XCTest
@testable import MLX_Server

final class ModelBackedQuantizationTests: XCTestCase {
    func testQuantizedLookupRoundTripPreservesRealModelCache() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)
        let input = quantizationPrompt()
        let prepared = try await engine.prepare(input)

        let workingCache = try await generatePromptCache(
            engine: engine,
            prepared: prepared,
            maxTokens: 1
        )

        let cache = TokenPrefixCache(
            memoryBudgetBytes: 1_000_000_000,
            quantizationConfig: .init(enabled: true, bits: 8, groupSize: 64, minTokens: 1)
        )
        cache.store(
            entryId: UUID(),
            kvCache: workingCache,
            cacheKey: prepared.tokens,
            modelId: "gemma"
        )

        let lease = cache.lookup(cacheKey: prepared.tokens, modelId: "gemma")
        let roundTripped = try XCTUnwrap(lease.kvCache)

        XCTAssertTrue(lease.isHit)
        XCTAssertFalse(roundTripped.isEmpty)
        XCTAssertFalse(roundTripped.contains { $0 is QuantizedKVCache })
        XCTAssertEqual(workingCache.count, roundTripped.count)

        for (original, returned) in zip(workingCache, roundTripped) {
            XCTAssertEqual(original.offset, returned.offset)
            XCTAssertEqual(original.state.count, returned.state.count)
            for (lhs, rhs) in zip(original.state, returned.state) {
                XCTAssertEqual(lhs.shape, rhs.shape)
            }
        }
    }

    func testQuantizedCacheHitProducesUsableDeterministicResponseAndAdvancesCacheLikeUnquantizedHit() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)
        let input = quantizationPrompt()
        let prepared = try await engine.prepare(input)

        let promptCache = try await generatePromptCache(
            engine: engine,
            prepared: prepared,
            maxTokens: 1
        )

        let unquantizedCache = TokenPrefixCache(
            memoryBudgetBytes: 1_000_000_000,
            quantizationConfig: .default
        )
        let quantizedCache = TokenPrefixCache(
            memoryBudgetBytes: 1_000_000_000,
            quantizationConfig: .init(enabled: true, bits: 8, groupSize: 64, minTokens: 1)
        )

        unquantizedCache.store(
            entryId: UUID(),
            kvCache: promptCache,
            cacheKey: prepared.tokens,
            modelId: "gemma"
        )
        quantizedCache.store(
            entryId: UUID(),
            kvCache: promptCache,
            cacheKey: prepared.tokens,
            modelId: "gemma"
        )

        let unquantizedLease = unquantizedCache.lookup(cacheKey: prepared.tokens, modelId: "gemma")
        let quantizedLease = quantizedCache.lookup(cacheKey: prepared.tokens, modelId: "gemma")

        XCTAssertTrue(unquantizedLease.isHit)
        XCTAssertTrue(quantizedLease.isHit)
        XCTAssertEqual(unquantizedLease.matchedTokenCount, prepared.tokens.count)
        XCTAssertEqual(quantizedLease.matchedTokenCount, prepared.tokens.count)

        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let unquantizedHandle = try await engine.stream(
            InferenceEngine.InferenceRequest(
                input: prepared.lmInput,
                tokens: prepared.tokens,
                parameters: parameters,
                cachedKV: unquantizedLease.kvCache,
                cachedTokenCount: unquantizedLease.matchedTokenCount
            ),
            cancellation: CancellationToken()
        )

        let unquantizedText = await collectText(unquantizedHandle.stream)
        XCTAssertFalse(unquantizedText.isEmpty)

        let quantizedHandle = try await engine.stream(
            InferenceEngine.InferenceRequest(
                input: prepared.lmInput,
                tokens: prepared.tokens,
                parameters: parameters,
                cachedKV: quantizedLease.kvCache,
                cachedTokenCount: quantizedLease.matchedTokenCount
            ),
            cancellation: CancellationToken()
        )
        let quantizedText = await collectText(quantizedHandle.stream)
        XCTAssertFalse(quantizedText.isEmpty)

        XCTAssertEqual(unquantizedHandle.workingCache.count, quantizedHandle.workingCache.count)
        for (lhs, rhs) in zip(unquantizedHandle.workingCache, quantizedHandle.workingCache) {
            XCTAssertLessThanOrEqual(abs(lhs.offset - rhs.offset), 1)
            XCTAssertEqual(lhs.state.count, rhs.state.count)
            for (lhsState, rhsState) in zip(lhs.state, rhs.state) {
                XCTAssertEqual(lhsState.shape.count, rhsState.shape.count)
                if lhsState.shape.count == 4 {
                    XCTAssertEqual(lhsState.shape[0], rhsState.shape[0])
                    XCTAssertEqual(lhsState.shape[1], rhsState.shape[1])
                    XCTAssertLessThanOrEqual(abs(lhsState.shape[2] - rhsState.shape[2]), 1)
                    XCTAssertEqual(lhsState.shape[3], rhsState.shape[3])
                } else {
                    XCTAssertEqual(lhsState.shape, rhsState.shape)
                }
            }
        }
    }

    func testPreferencesIntegrationWithQuantization() throws {
        Preferences.kvQuantizationEnabled = true
        Preferences.kvQuantizationBits = 8

        XCTAssertTrue(Preferences.kvQuantizationEnabled)
        XCTAssertEqual(Preferences.kvQuantizationBits, 8)

        Preferences.kvQuantizationBits = 2
        XCTAssertGreaterThanOrEqual(Preferences.kvQuantizationBits, 4)

        Preferences.kvQuantizationBits = 32
        XCTAssertLessThanOrEqual(Preferences.kvQuantizationBits, 16)

        Preferences.kvQuantizationEnabled = false
        Preferences.kvQuantizationBits = 8
    }

    private func quantizationPrompt() -> UserInput {
        UserInput(
            prompt: .chat([
                Chat.Message(role: .system, content: "You are terse and deterministic."),
                Chat.Message(role: .user, content: String(repeating: "cache reuse test ", count: 48))
            ]),
            images: [],
            videos: [],
            tools: nil
        )
    }

    private func generatePromptCache(
        engine: InferenceEngine,
        prepared: InferenceEngine.PreparedInference,
        maxTokens: Int
    ) async throws -> [KVCache] {
        let handle = try await engine.stream(
            InferenceEngine.InferenceRequest(
                input: prepared.lmInput,
                tokens: prepared.tokens,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
                cachedKV: nil,
                cachedTokenCount: 0
            ),
            cancellation: CancellationToken()
        )

        _ = await collectText(handle.stream)
        trimCacheToPrompt(handle.workingCache, promptTokenCount: prepared.tokens.count)
        return handle.workingCache
    }

    private func collectText(_ stream: AsyncStream<Generation>) async -> String {
        var text = ""
        for await generation in stream {
            if case .chunk(let chunk) = generation {
                text += chunk
            }
        }
        return text
    }

    private func trimCacheToPrompt(_ cache: [KVCache], promptTokenCount: Int) {
        for layer in cache {
            let excess = layer.offset - promptTokenCount
            if excess > 0 {
                XCTAssertTrue(layer.isTrimmable)
                XCTAssertEqual(layer.trim(excess), excess)
            }
        }
    }

    private func localGemmaContainer() async throws -> ModelContainer {
        try await LocalGemmaFixture.shared.container()
    }
}

// MARK: - LocalGemmaFixture

private actor LocalGemmaFixture {
    static let shared = LocalGemmaFixture()

    private var task: Task<ModelContainer, Error>?

    func container() async throws -> ModelContainer {
        if let task {
            return try await task.value
        }

        guard let config = ModelConfig.resolve("gemma") else {
            throw XCTSkip("Gemma model config is unavailable")
        }
        guard let localDir = LocalModelResolver.resolve(repoId: config.repoId) else {
            throw XCTSkip("Local gemma cache is unavailable")
        }

        let loadTask = Task<ModelContainer, Error> {
            let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            let hub = HubApi(downloadBase: cachesDir, cache: nil)
            return try await VLMModelFactory.shared.loadContainer(
                hub: hub,
                configuration: ModelConfiguration(directory: localDir),
                progressHandler: { _ in }
            )
        }
        task = loadTask

        do {
            return try await loadTask.value
        } catch {
            task = nil
            throw error
        }
    }
}

