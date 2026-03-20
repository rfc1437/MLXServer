import Foundation
import XCTest
@testable import MLX_Server

final class APIServerRewriteTests: XCTestCase {
    func testQwenNonStreamingChatCompletionCachesAndReusesPrompt() async throws {
        let harness = try await makeHarness(initialModelId: "qwen")
        defer { harness.stop() }

        let lookups = LookupEventCollector()
        APIServer.debugLookupEventHandler = { event in
            Task {
                await lookups.record(event)
            }
        }
        defer {
            APIServer.debugLookupEventHandler = nil
        }

        let request = APIChatCompletionRequest(
            model: "qwen",
            messages: [
                APIChatMessage(role: "user", content: .text("Reply with exactly one short word."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 1,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let firstResponse = try await sendChatCompletion(request, port: harness.port)
        XCTAssertEqual(firstResponse.choices.count, 1)

        try await waitUntil(timeoutSeconds: 5) {
            let snapshot = TokenPrefixCache.shared.snapshot()
            return snapshot.totalEntries > 0 && snapshot.entries.allSatisfy { $0.modelId == "qwen" }
        }

        let firstSnapshot = TokenPrefixCache.shared.snapshot()
        _ = try await sendChatCompletion(request, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            let events = await lookups.events()
            return events.count >= 2 && TokenPrefixCache.shared.snapshot().totalHits > firstSnapshot.totalHits
        }

        let secondSnapshot = TokenPrefixCache.shared.snapshot()
        let events = await lookups.events()
        let secondLookup = try XCTUnwrap(events.last)
        XCTAssertGreaterThan(secondSnapshot.totalHits, firstSnapshot.totalHits)
        XCTAssertTrue(secondLookup.isHit)
        XCTAssertGreaterThan(secondLookup.matchedTokenCount, 0)
    }

    func testHealthAndModelsEndpointsReturnExpectedPayloads() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let health = try await sendRawRequest(path: "/health", port: harness.port)
        XCTAssertEqual(health.statusCode, 200)
        XCTAssertEqual(health.body, #"{"status":"ok"}"#)

        let models = try await sendModelsRequest(port: harness.port)
        XCTAssertFalse(models.data.isEmpty)
        XCTAssertTrue(models.data.contains { $0.id == ModelConfig.default.repoId })
        XCTAssertTrue(models.data.allSatisfy { $0.context_window != nil })
    }

    func testNonStreamingChatCompletionUsesStatelessServerPathAndCachesPrompt() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Reply with exactly one short word."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 1,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let firstResponse = try await sendChatCompletion(request, port: harness.port)
        XCTAssertEqual(firstResponse.choices.count, 1)
        XCTAssertEqual(firstResponse.choices[0].message.role, "assistant")
        XCTAssertGreaterThan(firstResponse.usage.prompt_tokens, 0)
        XCTAssertGreaterThanOrEqual(firstResponse.usage.completion_tokens, 0)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > 0
        }
        let firstSnapshot = TokenPrefixCache.shared.snapshot()
        let firstLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(firstSnapshot.totalEntries, 0)
        XCTAssertGreaterThan(firstLiveSnapshot.prefillTokensPerSecond, 0)
        XCTAssertGreaterThan(firstLiveSnapshot.timeToFirstToken, 0)

        _ = try await sendChatCompletion(request, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > firstSnapshot.totalHits
        }
        let secondSnapshot = TokenPrefixCache.shared.snapshot()
        let secondLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(secondSnapshot.totalHits, firstSnapshot.totalHits)
        XCTAssertGreaterThan(secondLiveSnapshot.totalCacheReusePromptTokens, firstLiveSnapshot.totalCacheReusePromptTokens)
        XCTAssertGreaterThan(secondLiveSnapshot.cacheMatchDepth, 0)
    }

    func testSecondIdenticalRequestIsFullCacheHitWithZeroRebuiltPromptTokens() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let lookups = LookupEventCollector()
        APIServer.debugLookupEventHandler = { event in
            Task {
                await lookups.record(event)
            }
        }
        defer {
            APIServer.debugLookupEventHandler = nil
        }

        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer with one word: ocean."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(request, port: harness.port)
        _ = try await sendChatCompletion(request, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            let events = await lookups.events()
            return events.count >= 2
        }

        let events = await lookups.events()
        let secondLookup = try XCTUnwrap(events.last)
        XCTAssertTrue(secondLookup.isHit)
        XCTAssertEqual(secondLookup.matchedTokenCount, secondLookup.promptTokenCount)
    }

    func testSingleTurnContinuationProducesPartialCacheHit() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: sun."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let firstStream = try await sendStreamingChatCompletion(firstRequest, port: harness.port)
        XCTAssertFalse(firstStream.content.isEmpty)

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: sun."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: .text(firstStream.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: moon."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(secondRequest, port: harness.port)

        let live = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(live.currentCacheMatchedPromptTokens, 0)
        XCTAssertGreaterThan(live.currentCacheRebuiltPromptTokens, 0)
    }

    func testSameSystemPromptDifferentUserMessageReusesSystemPrefix() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let lookups = LookupEventCollector()
        APIServer.debugLookupEventHandler = { event in
            Task {
                await lookups.record(event)
            }
        }
        defer {
            APIServer.debugLookupEventHandler = nil
        }

        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Respond with one word for cat."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Respond with one word for dog."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(firstRequest, port: harness.port)
        _ = try await sendChatCompletion(secondRequest, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            let events = await lookups.events()
            return events.count >= 2
        }

        let events = await lookups.events()
        let secondLookup = try XCTUnwrap(events.last)
        XCTAssertEqual(secondLookup.modelId, "gemma")
        XCTAssertGreaterThan(secondLookup.promptTokenCount, 0)
        XCTAssertTrue(secondLookup.isHit)
        XCTAssertGreaterThan(secondLookup.matchedTokenCount, 0)
        XCTAssertLessThan(secondLookup.matchedTokenCount, secondLookup.promptTokenCount)
    }

    func testServerStoredCacheIsDirectlyReusableForSameSystemDifferentUserPrompt() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Respond with one word for cat."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(firstRequest, port: harness.port)

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Respond with one word for dog."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let modelContainer = await MainActor.run { harness.modelManager.modelContainer }
        let container = try XCTUnwrap(modelContainer)
        let engine = InferenceEngine(container: container)
        let preparedPrompt = PromptBuilder.build(
            from: secondRequest,
            modelId: ModelConfig.default.repoId,
            thinkingEnabled: Preferences.enableThinking
        )
        let preparedInference = try await engine.prepare(preparedPrompt.userInput)

        let lease = TokenPrefixCache.shared.lookup(cacheKey: preparedInference.tokens, modelId: "gemma")

        XCTAssertTrue(lease.isHit)
        XCTAssertGreaterThan(lease.matchedTokenCount, 0)
    }

    func testDifferentSystemPromptDoesNotProduceFalseCacheHit() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("System Alpha Unique Tokens"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: tree."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("Completely Different Beta Markers"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: tree."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(firstRequest, port: harness.port)
        let before = TokenPrefixCache.shared.snapshot()
        _ = try await sendChatCompletion(secondRequest, port: harness.port)

        let after = TokenPrefixCache.shared.snapshot()
        let live = LiveCounters.shared.snapshot()
        XCTAssertEqual(after.totalHits, before.totalHits)
        XCTAssertEqual(live.currentCacheMatchedPromptTokens, 0)
    }

    func testIdleUnloadReloadInvalidatesCacheAndServesFreshRequest() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        Preferences.lastModelId = "gemma"
        let request = APIChatCompletionRequest(
            model: nil,
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: cloud."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(request, port: harness.port)
        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > 0
        }

        await MainActor.run {
            harness.modelManager.unloadModel()
        }
        let wasReadyAfterUnload = await MainActor.run { harness.modelManager.isReady }
        XCTAssertFalse(wasReadyAfterUnload)

        let before = TokenPrefixCache.shared.snapshot()
        let response = try await sendChatCompletion(request, port: harness.port)
        XCTAssertEqual(response.choices.count, 1)
        let isReadyAfterReload = await MainActor.run { harness.modelManager.isReady }
        XCTAssertTrue(isReadyAfterReload)

        let after = TokenPrefixCache.shared.snapshot()
        let live = LiveCounters.shared.snapshot()
        XCTAssertEqual(after.totalHits, before.totalHits)
        XCTAssertEqual(live.currentCacheMatchedPromptTokens, 0)
    }

    func testRequestModelFieldSwapsFromGemmaToQwenAndInvalidatesGemmaCache() async throws {
        let harness = try await makeHarness(initialModelId: "gemma")
        defer { harness.stop() }

        let lookups = LookupEventCollector()
        APIServer.debugLookupEventHandler = { event in
            Task {
                await lookups.record(event)
            }
        }
        defer {
            APIServer.debugLookupEventHandler = nil
        }

        let gemmaRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer with one word: river."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(gemmaRequest, port: harness.port)
        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().entries.contains(where: { $0.modelId == "gemma" })
        }

        let qwenRequest = APIChatCompletionRequest(
            model: "qwen",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer with one word: river."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        _ = try await sendChatCompletion(qwenRequest, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            let snapshot = TokenPrefixCache.shared.snapshot()
            let modelId = await MainActor.run { harness.modelManager.currentModel?.id }
            return modelId == "qwen"
                && !snapshot.entries.isEmpty
                && snapshot.entries.allSatisfy { $0.modelId == "qwen" }
        }

        let afterSwapSnapshot = TokenPrefixCache.shared.snapshot()
        let afterSwapEvents = await lookups.events()
        let firstQwenLookup = try XCTUnwrap(afterSwapEvents.last)
        XCTAssertTrue(afterSwapSnapshot.entries.allSatisfy { $0.modelId == "qwen" })
        XCTAssertFalse(firstQwenLookup.isHit)
        XCTAssertEqual(firstQwenLookup.matchedTokenCount, 0)

        _ = try await sendChatCompletion(qwenRequest, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            let events = await lookups.events()
            return events.count >= 3 && TokenPrefixCache.shared.snapshot().totalHits > afterSwapSnapshot.totalHits
        }

        let finalSnapshot = TokenPrefixCache.shared.snapshot()
        let finalEvents = await lookups.events()
        let secondQwenLookup = try XCTUnwrap(finalEvents.last)
        XCTAssertGreaterThan(finalSnapshot.totalHits, afterSwapSnapshot.totalHits)
        XCTAssertTrue(secondQwenLookup.isHit)
        XCTAssertGreaterThan(secondQwenLookup.matchedTokenCount, 0)
    }

    func testStreamingChatCompletionReusesCacheAcrossThreeProgressivelyLongerTurns() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is the sky on a clear day?"), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 3,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let firstStream = try await sendStreamingChatCompletion(firstRequest, port: harness.port)
        XCTAssertEqual(firstStream.roleDeltaCount, 1)
        XCTAssertTrue(firstStream.sawDone)
        XCTAssertEqual(firstStream.finalFinishReason, "stop")
        XCTAssertGreaterThan(firstStream.usage?.prompt_tokens ?? 0, 0)
        XCTAssertFalse(firstStream.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > 0
        }
        let firstSnapshot = TokenPrefixCache.shared.snapshot()
        let firstLiveSnapshot = LiveCounters.shared.snapshot()

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is the sky on a clear day?"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: .text(firstStream.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is grass?"), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 3,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let secondStream = try await sendStreamingChatCompletion(secondRequest, port: harness.port)
        XCTAssertEqual(secondStream.roleDeltaCount, 1)
        XCTAssertTrue(secondStream.sawDone)
        XCTAssertEqual(secondStream.finalFinishReason, "stop")
        XCTAssertFalse(secondStream.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > firstSnapshot.totalHits
        }
        let secondSnapshot = TokenPrefixCache.shared.snapshot()
        let secondLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(secondSnapshot.totalHits, firstSnapshot.totalHits)
        XCTAssertGreaterThan(secondLiveSnapshot.totalCacheReusePromptTokens, firstLiveSnapshot.totalCacheReusePromptTokens)

        let thirdRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is the sky on a clear day?"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: .text(firstStream.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is grass?"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: .text(secondStream.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Answer in one word: what color is snow?"), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 3,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let thirdStream = try await sendStreamingChatCompletion(thirdRequest, port: harness.port)
        XCTAssertEqual(thirdStream.roleDeltaCount, 1)
        XCTAssertTrue(thirdStream.sawDone)
        XCTAssertEqual(thirdStream.finalFinishReason, "stop")
        XCTAssertFalse(thirdStream.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > secondSnapshot.totalHits
        }
        let thirdSnapshot = TokenPrefixCache.shared.snapshot()
        let thirdLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(thirdSnapshot.totalHits, secondSnapshot.totalHits)
        XCTAssertGreaterThan(thirdLiveSnapshot.totalCacheReusePromptTokens, secondLiveSnapshot.totalCacheReusePromptTokens)
    }

    func testStreamingChatCompletionReusesCacheAcrossToolBoundary() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let tools = [mockWeatherTool]
        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("You must call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 48,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let toolCallStream = try await sendStreamingChatCompletion(firstRequest, port: harness.port)
        XCTAssertEqual(toolCallStream.roleDeltaCount, 1)
        XCTAssertTrue(toolCallStream.sawDone)
        XCTAssertEqual(toolCallStream.finalFinishReason, "tool_calls")
        let toolCall = try XCTUnwrap(toolCallStream.toolCalls.first)
        XCTAssertEqual(toolCall.function.name, "weather")

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > 0
        }
        let afterToolCallSnapshot = TokenPrefixCache.shared.snapshot()
        let afterToolCallLiveSnapshot = LiveCounters.shared.snapshot()

        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("You must call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [toolCall], tool_call_id: nil),
                APIChatMessage(role: "tool", content: .text("{\"city\":\"Berlin\",\"temperature_c\":19,\"condition\":\"sunny\"}"), name: nil, tool_calls: nil, tool_call_id: toolCall.id)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 16,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let directAnswerStream = try await sendStreamingChatCompletion(secondRequest, port: harness.port)
        XCTAssertEqual(directAnswerStream.roleDeltaCount, 1)
        XCTAssertTrue(directAnswerStream.sawDone)
        XCTAssertEqual(directAnswerStream.finalFinishReason, "stop")
        XCTAssertTrue(directAnswerStream.toolCalls.isEmpty)
        XCTAssertFalse(directAnswerStream.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > afterToolCallSnapshot.totalHits
        }
        let afterDirectAnswerSnapshot = TokenPrefixCache.shared.snapshot()
        let afterDirectAnswerLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(afterDirectAnswerSnapshot.totalHits, afterToolCallSnapshot.totalHits)
        XCTAssertGreaterThan(afterDirectAnswerLiveSnapshot.totalCacheReusePromptTokens, afterToolCallLiveSnapshot.totalCacheReusePromptTokens)

        let thirdRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("You must call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [toolCall], tool_call_id: nil),
                APIChatMessage(role: "tool", content: .text("{\"city\":\"Berlin\",\"temperature_c\":19,\"condition\":\"sunny\"}"), name: nil, tool_calls: nil, tool_call_id: toolCall.id),
                APIChatMessage(role: "assistant", content: .text(directAnswerStream.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Now compress that answer to two words."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 8,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let thirdStream = try await sendStreamingChatCompletion(thirdRequest, port: harness.port)
        XCTAssertEqual(thirdStream.roleDeltaCount, 1)
        XCTAssertTrue(thirdStream.sawDone)
        XCTAssertEqual(thirdStream.finalFinishReason, "stop")
        XCTAssertFalse(thirdStream.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > afterDirectAnswerSnapshot.totalHits
        }
        let finalSnapshot = TokenPrefixCache.shared.snapshot()
        let finalLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(finalSnapshot.totalHits, afterDirectAnswerSnapshot.totalHits)
        XCTAssertGreaterThan(finalLiveSnapshot.totalCacheReusePromptTokens, afterDirectAnswerLiveSnapshot.totalCacheReusePromptTokens)
    }

    func testStreamingChatCompletionReusesCacheAcrossMultipleToolTurns() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let tools = [mockWeatherTool]
        let berlinRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 48,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let firstToolTurn = try await sendStreamingChatCompletion(berlinRequest, port: harness.port)
        XCTAssertEqual(firstToolTurn.finalFinishReason, "tool_calls")
        let berlinToolCall = try XCTUnwrap(firstToolTurn.toolCalls.first)
        XCTAssertEqual(berlinToolCall.function.name, "weather")

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > 0
        }
        let firstSnapshot = TokenPrefixCache.shared.snapshot()
        let firstLiveSnapshot = LiveCounters.shared.snapshot()

        let berlinToolResult = APIChatMessage(
            role: "tool",
            content: .text("{\"city\":\"Berlin\",\"temperature_c\":19,\"condition\":\"sunny\"}"),
            name: nil,
            tool_calls: nil,
            tool_call_id: berlinToolCall.id
        )

        let berlinAnswerRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [berlinToolCall], tool_call_id: nil),
                berlinToolResult
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 16,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let berlinAnswer = try await sendStreamingChatCompletion(berlinAnswerRequest, port: harness.port)
        XCTAssertEqual(berlinAnswer.finalFinishReason, "stop")
        XCTAssertFalse(berlinAnswer.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > firstSnapshot.totalHits
        }
        let secondSnapshot = TokenPrefixCache.shared.snapshot()
        let secondLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(secondSnapshot.totalHits, firstSnapshot.totalHits)
        XCTAssertGreaterThan(secondLiveSnapshot.totalCacheReusePromptTokens, firstLiveSnapshot.totalCacheReusePromptTokens)

        let parisToolTurnRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [berlinToolCall], tool_call_id: nil),
                berlinToolResult,
                APIChatMessage(role: "assistant", content: .text(berlinAnswer.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Now call the weather tool for Paris. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 48,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let secondToolTurn = try await sendStreamingChatCompletion(parisToolTurnRequest, port: harness.port)
        XCTAssertEqual(secondToolTurn.finalFinishReason, "tool_calls")
        let parisToolCall = try XCTUnwrap(secondToolTurn.toolCalls.first)
        XCTAssertEqual(parisToolCall.function.name, "weather")

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > secondSnapshot.totalHits
        }
        let thirdSnapshot = TokenPrefixCache.shared.snapshot()
        let thirdLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(thirdSnapshot.totalHits, secondSnapshot.totalHits)
        XCTAssertGreaterThan(thirdLiveSnapshot.totalCacheReusePromptTokens, secondLiveSnapshot.totalCacheReusePromptTokens)

        let parisAnswerRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [berlinToolCall], tool_call_id: nil),
                berlinToolResult,
                APIChatMessage(role: "assistant", content: .text(berlinAnswer.content), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Now call the weather tool for Paris. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: nil, name: nil, tool_calls: [parisToolCall], tool_call_id: nil),
                APIChatMessage(role: "tool", content: .text("{\"city\":\"Paris\",\"temperature_c\":21,\"condition\":\"clear\"}"), name: nil, tool_calls: nil, tool_call_id: parisToolCall.id)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 16,
            stream: true,
            stop: nil,
            tools: tools,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let parisAnswer = try await sendStreamingChatCompletion(parisAnswerRequest, port: harness.port)
        XCTAssertEqual(parisAnswer.finalFinishReason, "stop")
        XCTAssertTrue(parisAnswer.toolCalls.isEmpty)
        XCTAssertFalse(parisAnswer.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > thirdSnapshot.totalHits
        }
        let fourthSnapshot = TokenPrefixCache.shared.snapshot()
        let fourthLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(fourthSnapshot.totalHits, thirdSnapshot.totalHits)
        XCTAssertGreaterThan(fourthLiveSnapshot.totalCacheReusePromptTokens, thirdLiveSnapshot.totalCacheReusePromptTokens)
    }

    func testStreamingDisconnectStoresPromptCacheForReuse() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Count from one to twenty with commas, using many tokens."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 64,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let initialSnapshot = TokenPrefixCache.shared.snapshot()
        try await cancelStreamingChatCompletionAfterFirstContent(request, port: harness.port)

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalEntries > initialSnapshot.totalEntries
        }
        let afterDisconnectSnapshot = TokenPrefixCache.shared.snapshot()
        let afterDisconnectLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(afterDisconnectSnapshot.totalEntries, initialSnapshot.totalEntries)
        XCTAssertGreaterThan(afterDisconnectLiveSnapshot.totalDisconnects, 0)

        _ = try await sendChatCompletion(
            APIChatCompletionRequest(
                model: request.model,
                messages: request.messages,
                temperature: request.temperature,
                top_p: request.top_p,
                max_tokens: 8,
                stream: false,
                stop: request.stop,
                tools: request.tools,
                tool_choice: request.tool_choice,
                frequency_penalty: request.frequency_penalty,
                presence_penalty: request.presence_penalty,
                n: request.n
            ),
            port: harness.port
        )

        try await waitUntil(timeoutSeconds: 5) {
            TokenPrefixCache.shared.snapshot().totalHits > afterDisconnectSnapshot.totalHits
        }
        let finalSnapshot = TokenPrefixCache.shared.snapshot()
        let finalLiveSnapshot = LiveCounters.shared.snapshot()
        XCTAssertGreaterThan(finalSnapshot.totalHits, afterDisconnectSnapshot.totalHits)
        XCTAssertGreaterThan(finalLiveSnapshot.totalCacheReusePromptTokens, afterDisconnectLiveSnapshot.totalCacheReusePromptTokens)
    }

    func testStreamingDisconnectStopsServerWorkWithinTwoHundredMilliseconds() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Count from one to fifty with commas, using many tokens."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 128,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let url = URL(string: "http://127.0.0.1:\(harness.port)/v1/chat/completions")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let observer = StreamCancellationObserver()
        let session = URLSession(configuration: .ephemeral)
        let baselineDisconnects = LiveCounters.shared.snapshot().totalDisconnects
        let task = Task {
            let (bytes, response) = try await session.bytes(for: urlRequest)
            let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
            XCTAssertEqual(httpResponse.statusCode, 200)

            for try await line in bytes.lines {
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst(6))
                if payload == "[DONE]" {
                    break
                }
                guard let data = payload.data(using: .utf8) else { continue }
                let chunk = try JSONDecoder().decode(APIChatCompletionChunk.self, from: data)
                if let deltaContent = chunk.choices.first?.delta.content, !deltaContent.isEmpty {
                    await observer.markFirstContentSeen()
                    try await Task.sleep(nanoseconds: 30_000_000_000)
                }
            }
        }

        try await waitUntil(timeoutSeconds: 10) {
            await observer.hasSeenFirstContent
        }

        let disconnectStartedAt = Date()
        session.invalidateAndCancel()
        task.cancel()

        try await waitUntil(timeoutSeconds: 5, intervalNanoseconds: 10_000_000) {
            let snapshot = LiveCounters.shared.snapshot()
            return snapshot.totalDisconnects > baselineDisconnects && snapshot.activeRequests == 0
        }

        _ = try? await task.value
        let elapsed = Date().timeIntervalSince(disconnectStartedAt)
        XCTAssertLessThan(elapsed, 0.2)
    }

    func testRepeatedStreamingDisconnectsDoNotBreakSubsequentGeneration() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Count from one to forty with commas, using many tokens."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 96,
            stream: true,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        for expectedDisconnectCount in 1...3 {
            try await cancelStreamingChatCompletionAfterFirstContentAndWaitForServerDisconnect(
                request,
                port: harness.port,
                expectedDisconnectCount: expectedDisconnectCount
            )

            let liveSnapshot = LiveCounters.shared.snapshot()
            XCTAssertEqual(liveSnapshot.totalDisconnects, expectedDisconnectCount)
            XCTAssertEqual(liveSnapshot.activeRequests, 0)
        }

        let recoveryRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Reply with exactly one short word."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: 0,
            top_p: 1,
            max_tokens: 2,
            stream: false,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let response = try await sendChatCompletion(recoveryRequest, port: harness.port)
        XCTAssertEqual(response.choices.count, 1)
        XCTAssertEqual(response.choices[0].message.role, "assistant")
        XCTAssertFalse((response.choices[0].message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    func testStreamingToolCallChunksArriveInOpenAICompatibleOrder() async throws {
        let harness = try await makeHarness()
        defer { harness.stop() }

        let detailed = try await sendStreamingChatCompletionDetailed(
            APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "user", content: .text("Call the weather tool for Berlin. Do not answer directly."), name: nil, tool_calls: nil, tool_call_id: nil)
                ],
                temperature: 0,
                top_p: 1,
                max_tokens: 48,
                stream: true,
                stop: nil,
                tools: [mockWeatherTool],
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            port: harness.port
        )

        XCTAssertTrue(detailed.sawDone)
        XCTAssertFalse(detailed.events.isEmpty)

        let firstEvent = try XCTUnwrap(detailed.events.first)
        XCTAssertEqual(firstEvent.kind, .role)
        XCTAssertEqual(firstEvent.role, "assistant")

        let toolEventIndices = detailed.events.enumerated().compactMap { index, event in
            event.kind == .toolCall ? index : nil
        }
        XCTAssertFalse(toolEventIndices.isEmpty)

        let finalIndex = try XCTUnwrap(detailed.events.lastIndex(where: { $0.kind == .final }))
        XCTAssertEqual(finalIndex, detailed.events.count - 1)

        for toolIndex in toolEventIndices {
            XCTAssertLessThan(toolIndex, finalIndex)
        }

        let finalEvent = detailed.events[finalIndex]
        XCTAssertEqual(finalEvent.finishReason, "tool_calls")
        XCTAssertNotNil(finalEvent.usage)

        let roleEventCount = detailed.events.filter { $0.kind == .role }.count
        XCTAssertEqual(roleEventCount, 1)
    }

    private var mockWeatherTool: APIToolDefinition {
        APIToolDefinition(
            type: "function",
            function: APIFunctionDefinition(
                name: "weather",
                description: "Look up weather for a city.",
                parameters: [
                    "type": AnyCodable("object"),
                    "properties": AnyCodable([
                        "city": [
                            "type": "string",
                            "description": "City name"
                        ]
                    ]),
                    "required": AnyCodable(["city"])
                ]
            )
        )
    }

    private func makeHarness(initialModelId: String = "gemma") async throws -> TestHarness {
        let modelManager = await MainActor.run { ModelManager() }
        let config = try XCTUnwrap(ModelConfig.resolve(initialModelId))

        LiveCounters.shared.reset()
        TokenPrefixCache.shared.reset()
        await modelManager.loadModel(config)
        let isReady = await MainActor.run { modelManager.isReady }
        XCTAssertTrue(isReady)

        let server = await MainActor.run { APIServer() }
        let port = UInt16.random(in: 20_000...40_000)
        await MainActor.run {
            server.start(modelManager: modelManager, port: Int(port))
        }

        try await waitUntil(timeoutSeconds: 5) {
            await MainActor.run { server.isRunning }
        }

        return TestHarness(server: server, modelManager: modelManager, port: port)
    }

    private func sendChatCompletion(_ request: APIChatCompletionRequest, port: UInt16) async throws -> APIChatCompletionResponse {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(httpResponse.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
        return try JSONDecoder().decode(APIChatCompletionResponse.self, from: data)
    }

    private func sendModelsRequest(port: UInt16) async throws -> APIModelListResponse {
        let response = try await sendRawRequest(path: "/v1/models", port: port)
        XCTAssertEqual(response.statusCode, 200)
        return try JSONDecoder().decode(APIModelListResponse.self, from: response.bodyData)
    }

    private func sendRawRequest(path: String, port: UInt16) async throws -> (statusCode: Int, body: String, bodyData: Data) {
        let url = URL(string: "http://127.0.0.1:\(port)\(path)")!
        let (data, response) = try await URLSession.shared.data(from: url)
        let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
        return (httpResponse.statusCode, String(data: data, encoding: .utf8) ?? "", data)
    }

    private func sendStreamingChatCompletion(_ request: APIChatCompletionRequest, port: UInt16) async throws -> StreamingResult {
        let detailed = try await sendStreamingChatCompletionDetailed(request, port: port)
        return StreamingResult(
            roleDeltaCount: detailed.events.filter { $0.kind == .role }.count,
            content: detailed.events.compactMap(\ .content).joined(),
            toolCalls: detailed.events.flatMap(\ .toolCalls),
            finalFinishReason: detailed.events.last(where: { $0.kind == .final })?.finishReason,
            usage: detailed.events.last(where: { $0.kind == .final })?.usage,
            sawDone: detailed.sawDone
        )
    }

    private func sendStreamingChatCompletionDetailed(_ request: APIChatCompletionRequest, port: UInt16) async throws -> DetailedStreamingResult {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
        let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
        guard httpResponse.statusCode == 200 else {
            var body = ""
            for try await line in bytes.lines {
                body += line
            }
            XCTFail("Expected 200 response, got \(httpResponse.statusCode): \(body)")
            return DetailedStreamingResult(events: [], sawDone: false)
        }

        var events: [StreamingEvent] = []
        var sawDone = false

        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" {
                sawDone = true
                break
            }

            guard let data = payload.data(using: .utf8) else { continue }
            let chunk = try JSONDecoder().decode(APIChatCompletionChunk.self, from: data)
            let choice = chunk.choices.first
            if let delta = chunk.choices.first?.delta.role, delta == "assistant" {
                events.append(StreamingEvent(kind: .role, role: delta, content: nil, toolCalls: [], finishReason: nil, usage: nil))
            }
            if let deltaContent = chunk.choices.first?.delta.content {
                events.append(StreamingEvent(kind: .content, role: nil, content: deltaContent, toolCalls: [], finishReason: nil, usage: nil))
            }
            if let deltaToolCalls = chunk.choices.first?.delta.tool_calls {
                events.append(StreamingEvent(kind: .toolCall, role: nil, content: nil, toolCalls: deltaToolCalls, finishReason: nil, usage: nil))
            }
            if let finishReason = choice?.finish_reason {
                events.append(StreamingEvent(kind: .final, role: nil, content: nil, toolCalls: [], finishReason: finishReason, usage: chunk.usage))
            }
        }

        return DetailedStreamingResult(events: events, sawDone: sawDone)
    }

    private func cancelStreamingChatCompletionAfterFirstContent(_ request: APIChatCompletionRequest, port: UInt16) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let observer = StreamCancellationObserver()
        let session = URLSession(configuration: .ephemeral)
        let task = Task {
            let (bytes, response) = try await session.bytes(for: urlRequest)
            let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
            XCTAssertEqual(httpResponse.statusCode, 200)

            for try await line in bytes.lines {
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst(6))
                if payload == "[DONE]" {
                    break
                }
                guard let data = payload.data(using: .utf8) else { continue }
                let chunk = try JSONDecoder().decode(APIChatCompletionChunk.self, from: data)
                if let deltaContent = chunk.choices.first?.delta.content, !deltaContent.isEmpty {
                    await observer.markFirstContentSeen()
                    try await Task.sleep(nanoseconds: 30_000_000_000)
                }
            }
        }

        try await waitUntil(timeoutSeconds: 10) {
            await observer.hasSeenFirstContent
        }

        session.invalidateAndCancel()
        task.cancel()
        _ = try? await task.value
    }

    private func cancelStreamingChatCompletionAfterFirstContentAndWaitForServerDisconnect(
        _ request: APIChatCompletionRequest,
        port: UInt16,
        expectedDisconnectCount: Int
    ) async throws {
        try await cancelStreamingChatCompletionAfterFirstContent(request, port: port)

        try await waitUntil(timeoutSeconds: 5, intervalNanoseconds: 10_000_000) {
            let snapshot = LiveCounters.shared.snapshot()
            return snapshot.totalDisconnects >= expectedDisconnectCount && snapshot.activeRequests == 0
        }
    }

    private func waitUntil(
        timeoutSeconds: TimeInterval,
        intervalNanoseconds: UInt64 = 100_000_000,
        condition: @escaping () async -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if await condition() {
                return
            }
            try await Task.sleep(nanoseconds: intervalNanoseconds)
        }
        XCTFail("Condition not met before timeout")
    }
}

private actor StreamCancellationObserver {
    private var sawFirstContent = false

    func markFirstContentSeen() {
        sawFirstContent = true
    }

    var hasSeenFirstContent: Bool {
        sawFirstContent
    }
}

private actor LookupEventCollector {
    private var recorded: [APIServer.DebugLookupEvent] = []

    func record(_ event: APIServer.DebugLookupEvent) {
        recorded.append(event)
    }

    func events() -> [APIServer.DebugLookupEvent] {
        recorded
    }
}

private struct DetailedStreamingResult {
    let events: [StreamingEvent]
    let sawDone: Bool
}

private struct StreamingEvent {
    enum Kind {
        case role
        case content
        case toolCall
        case final
    }

    let kind: Kind
    let role: String?
    let content: String?
    let toolCalls: [APIToolCall]
    let finishReason: String?
    let usage: APIUsageInfo?
}

private struct StreamingResult {
    let roleDeltaCount: Int
    let content: String
    let toolCalls: [APIToolCall]
    let finalFinishReason: String?
    let usage: APIUsageInfo?
    let sawDone: Bool
}

private struct TestHarness {
    let server: APIServer
    let modelManager: ModelManager
    let port: UInt16

    func stop() {
        Task { @MainActor in
            server.stop()
            modelManager.unloadModel()
        }
        TokenPrefixCache.shared.reset()
    }
}