import Foundation
import Hub
import MLXLMCommon
import MLXVLM
import XCTest
@testable import MLX_Server

final class ModelBackedInferenceValidationTests: XCTestCase {
    private let onePixelPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lRyXWQAAAABJRU5ErkJggg=="

    func testPromptBuilderTokenizationMatchesLegacyShapingOnLocalGemma() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)
        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("You are concise."), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(
                    role: "user",
                    content: .parts([
                        APIContentPart(type: "text", text: "What is in this image?", image_url: nil),
                        APIContentPart(type: "image_url", text: nil, image_url: APIImageURL(url: "data:image/png;base64,\(onePixelPNGBase64)", detail: nil))
                    ]),
                    name: nil,
                    tool_calls: nil,
                    tool_call_id: nil
                )
            ],
            temperature: nil,
            top_p: nil,
            max_tokens: nil,
            stream: nil,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let prepared = PromptBuilder.build(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: false)
        let legacy = legacyBuild(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: false)

        let preparedInference = try await engine.prepare(prepared.userInput)
        let legacyInference = try await engine.prepare(legacy.userInput)

        XCTAssertEqual(preparedInference.tokens, legacyInference.tokens)
    }

    func testInferenceEngineMatchesChatSessionOnLocalGemma() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "user", content: .text("Say hello in one word."), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: nil,
            top_p: nil,
            max_tokens: nil,
            stream: nil,
            stop: nil,
            tools: nil,
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let prepared = PromptBuilder.build(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: true)
        let preparedInference = try await engine.prepare(prepared.userInput)
        let handle = try await engine.stream(
            InferenceEngine.InferenceRequest(
                input: preparedInference.lmInput,
                tokens: preparedInference.tokens,
                parameters: parameters,
                cachedKV: nil,
                cachedTokenCount: 0
            ),
            cancellation: CancellationToken()
        )

        let engineResult = await collectEngineOutput(handle.stream)

        let session = ChatSession(container, generateParameters: parameters)
        let sessionResult = try await collectSessionOutput(
            session.streamDetails(to: "Say hello in one word.", images: [], videos: [])
        )

        XCTAssertEqual(engineResult.text, sessionResult.text)
        XCTAssertEqual(engineResult.promptTokenCount, sessionResult.promptTokenCount)
    }

    func testTokenPrefixCacheFindsLCPHitForSameSystemDifferentUserOnLocalGemmaTokens() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)

        let first = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Respond with one word for cat."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )
        let second = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Respond with one word for dog."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )

        let firstPrepared = try await engine.prepare(first.userInput)
        let secondPrepared = try await engine.prepare(second.userInput)

        let cache = TokenPrefixCache(memoryBudgetBytes: 1_000_000, estimateBytesProvider: { _ in 1_024 })
        cache.store(entryId: UUID(), kvCache: [], cacheKey: firstPrepared.tokens, modelId: "gemma")

        let lease = cache.lookup(cacheKey: secondPrepared.tokens, modelId: "gemma")

        XCTAssertTrue(lease.isHit)
        XCTAssertGreaterThan(lease.matchedTokenCount, 0)
        XCTAssertLessThan(lease.matchedTokenCount, firstPrepared.tokens.count)
    }

    func testStoredLiveGemmaCacheSupportsSameSystemDifferentUserLCPReuse() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)

        let first = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Respond with one word for cat."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )
        let second = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("You are terse and literal."), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Respond with one word for dog."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )

        let firstPrepared = try await engine.prepare(first.userInput)
        let secondPrepared = try await engine.prepare(second.userInput)
        let handle = try await engine.stream(
            InferenceEngine.InferenceRequest(
                input: firstPrepared.lmInput,
                tokens: firstPrepared.tokens,
                parameters: GenerateParameters(maxTokens: 2, temperature: 0),
                cachedKV: nil,
                cachedTokenCount: 0
            ),
            cancellation: CancellationToken()
        )

        _ = await collectEngineOutput(handle.stream)
        trimCacheToPrompt(handle.workingCache, promptTokenCount: firstPrepared.tokens.count)

        let cache = TokenPrefixCache(memoryBudgetBytes: 1_000_000_000, estimateBytesProvider: { _ in 1_024 })
        cache.store(entryId: UUID(), kvCache: handle.workingCache, cacheKey: firstPrepared.tokens, modelId: "gemma")

        let lease = cache.lookup(cacheKey: secondPrepared.tokens, modelId: "gemma")

        XCTAssertTrue(lease.isHit)
        XCTAssertGreaterThan(lease.matchedTokenCount, 0)
        XCTAssertLessThan(lease.matchedTokenCount, firstPrepared.tokens.count)
    }

    func testTokenPrefixCacheCanFalseHitDifferentSystemPromptsOnRawGemmaTokens() async throws {
        let container = try await localGemmaContainer()
        let engine = InferenceEngine(container: container)

        let first = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("System Alpha Unique Tokens"), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Answer in one word: tree."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )
        let second = PromptBuilder.build(
            from: APIChatCompletionRequest(
                model: "gemma",
                messages: [
                    APIChatMessage(role: "system", content: .text("Completely Different Beta Markers"), name: nil, tool_calls: nil, tool_call_id: nil),
                    APIChatMessage(role: "user", content: .text("Answer in one word: tree."), name: nil, tool_calls: nil, tool_call_id: nil),
                ],
                temperature: nil,
                top_p: nil,
                max_tokens: nil,
                stream: nil,
                stop: nil,
                tools: nil,
                tool_choice: nil,
                frequency_penalty: nil,
                presence_penalty: nil,
                n: nil
            ),
            modelId: "mlx-community/gemma-3-4b-it-4bit",
            thinkingEnabled: true
        )

        let firstPrepared = try await engine.prepare(first.userInput)
        let secondPrepared = try await engine.prepare(second.userInput)

        let cache = TokenPrefixCache(memoryBudgetBytes: 1_000_000, estimateBytesProvider: { _ in 1_024 })
        cache.store(entryId: UUID(), kvCache: [], cacheKey: firstPrepared.tokens, modelId: "gemma")

        let lease = cache.lookup(cacheKey: secondPrepared.tokens, modelId: "gemma")

        XCTAssertFalse(lease.isHit)
    }

    private func localGemmaContainer() async throws -> ModelContainer {
        try await LocalGemmaFixture.shared.container()
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

    private func legacyBuild(
        from request: APIChatCompletionRequest,
        modelId: String,
        thinkingEnabled: Bool
    ) -> PromptBuilder.PreparedPrompt {
        var instructions = ""
        for msg in request.messages where msg.role == "system" {
            let text = msg.content?.textContent ?? ""
            if !text.isEmpty {
                if !instructions.isEmpty { instructions += "\n\n" }
                instructions += text
            }
        }

        if let tools = request.tools, !tools.isEmpty {
            let toolSystemPrompt = ToolPromptBuilder.buildSystemPrompt(tools: tools, modelId: modelId)
            if !instructions.isEmpty { instructions += "\n\n" }
            instructions += toolSystemPrompt
        }

        let isQwen = modelId.lowercased().contains("qwen")
        var chatMessages: [Chat.Message] = []
        var messageSignatures: [UInt64] = []
        var estimatedBytes = instructions.utf8.count
        var containsImages = false

        for msg in request.messages where msg.role != "system" {
            let role: Chat.Message.Role = switch msg.role {
            case "assistant": .assistant
            case "tool": .user
            default: .user
            }

            var text = msg.content?.textContent ?? ""
            if msg.role == "tool", !isQwen {
                text = "```tool_output\n\(text)\n```"
            }

            if msg.role == "assistant", let toolCalls = msg.tool_calls, !toolCalls.isEmpty {
                let formattedCalls = isQwen
                    ? ToolPromptBuilder.formatQwenToolCalls(toolCalls)
                    : ToolPromptBuilder.formatGemmaToolCalls(toolCalls)
                text = (text.isEmpty ? "" : text + "\n") + formattedCalls
            }

            let imageURLs = msg.content?.imageURLs ?? []
            var messageImages: [UserInput.Image] = []
            var messageImageBytes = 0
            for urlString in imageURLs {
                if let decoded = ImageDecoder.decode(urlString) {
                    messageImages.append(decoded.image)
                    messageImageBytes += decoded.estimatedBytes
                }
            }

            containsImages = containsImages || !messageImages.isEmpty
            chatMessages.append(Chat.Message(role: role, content: text, images: messageImages))
            messageSignatures.append(messageSignature(role: role, content: text, imageURLs: imageURLs))
            estimatedBytes += text.utf8.count + messageImageBytes
        }

        let additionalContext: [String: any Sendable]? = thinkingEnabled
            ? nil
            : ["enable_thinking": false]

        let allImages = chatMessages.flatMap(\.images)
        let allMessages = (instructions.isEmpty ? [] : [Chat.Message(role: .system, content: instructions)]) + chatMessages
        let userInput = UserInput(
            prompt: .chat(allMessages),
            images: allImages,
            videos: [],
            tools: nil,
            additionalContext: additionalContext
        )

        return PromptBuilder.PreparedPrompt(
            instructions: instructions,
            chatMessages: chatMessages,
            messageSignatures: messageSignatures,
            estimatedBytes: estimatedBytes,
            estimatedPromptTokens: (instructions.count + chatMessages.reduce(0) { $0 + $1.content.count }) * 10 / 35,
            containsImages: containsImages,
            additionalContext: additionalContext,
            userInput: userInput
        )
    }

    private func messageSignature(role: Chat.Message.Role, content: String, imageURLs: [String]) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037

        func mix(_ text: String) {
            for byte in text.utf8 {
                hash ^= UInt64(byte)
                hash &*= 1_099_511_628_211
            }
        }

        switch role {
        case .assistant:
            mix("assistant")
        case .system:
            mix("system")
        case .user:
            mix("user")
        @unknown default:
            mix("unknown")
        }
        mix("|")
        mix(content)
        for imageURL in imageURLs {
            mix("|")
            mix(imageURL)
        }

        return hash
    }

    private func collectEngineOutput(_ stream: AsyncStream<Generation>) async -> GenerationResult {
        var text = ""
        var promptTokenCount = 0
        for await generation in stream {
            switch generation {
            case .chunk(let chunk):
                text += chunk
            case .info(let info):
                promptTokenCount = info.promptTokenCount
            case .toolCall:
                break
            }
        }
        return GenerationResult(text: text, promptTokenCount: promptTokenCount)
    }

    private func collectSessionOutput(_ stream: AsyncThrowingStream<Generation, any Error>) async throws -> GenerationResult {
        var text = ""
        var promptTokenCount = 0
        for try await generation in stream {
            switch generation {
            case .chunk(let chunk):
                text += chunk
            case .info(let info):
                promptTokenCount = info.promptTokenCount
            case .toolCall:
                break
            }
        }
        return GenerationResult(text: text, promptTokenCount: promptTokenCount)
    }
}

private struct GenerationResult {
    let text: String
    let promptTokenCount: Int
}

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