import XCTest
import MLXLMCommon
@testable import MLX_Server

final class PromptBuilderTests: XCTestCase {
    func testBuildMatchesLegacyAPIServerShapingForGemma() {
        let toolCall = APIToolCall(
            id: "call_weather",
            function: APIFunctionCall(name: "weather", arguments: "{\"city\":\"Berlin\"}")
        )
        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("System 1"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "system", content: .text("System 2"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "assistant", content: .text("Let me check"), name: nil, tool_calls: [toolCall], tool_call_id: nil),
                APIChatMessage(
                    role: "tool",
                    content: .parts([
                        APIContentPart(type: "text", text: "{\"temp\":19}", image_url: nil),
                        APIContentPart(type: "image_url", text: nil, image_url: APIImageURL(url: TestImageFixtures.primaryDataURI, detail: nil))
                    ]),
                    name: nil,
                    tool_calls: nil,
                    tool_call_id: "call_weather"
                ),
                APIChatMessage(role: "user", content: .text("Thanks"), name: nil, tool_calls: nil, tool_call_id: nil)
            ],
            temperature: nil,
            top_p: nil,
            max_tokens: nil,
            stream: nil,
            stop: nil,
            tools: [
                APIToolDefinition(
                    type: "function",
                    function: APIFunctionDefinition(
                        name: "weather",
                        description: "Lookup weather",
                        parameters: ["type": AnyCodable("object")]
                    )
                )
            ],
            tool_choice: nil,
            frequency_penalty: nil,
            presence_penalty: nil,
            n: nil
        )

        let prepared = PromptBuilder.build(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: false)
        let legacy = legacyBuild(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: false)

        XCTAssertEqual(prepared.instructions, legacy.instructions)
        XCTAssertEqual(prepared.chatMessages.map { $0.role.roleLabel }, legacy.chatMessages.map { $0.role.roleLabel })
        XCTAssertEqual(prepared.chatMessages.map(\.content), legacy.chatMessages.map(\.content))
        XCTAssertEqual(prepared.chatMessages.map { $0.images.count }, legacy.chatMessages.map { $0.images.count })
        XCTAssertEqual(prepared.messageSignatures, legacy.messageSignatures)
        XCTAssertEqual(prepared.estimatedBytes, legacy.estimatedBytes)
        XCTAssertEqual(prepared.estimatedPromptTokens, legacy.estimatedPromptTokens)
        XCTAssertEqual(prepared.containsImages, legacy.containsImages)
        XCTAssertEqual(prepared.additionalContext?["enable_thinking"] as? Bool, legacy.additionalContext?["enable_thinking"] as? Bool)
    }

    func testBuildAggregatesInstructionsAndMessages() {
        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(role: "system", content: .text("Base system"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "system", content: .text("Extra system"), name: nil, tool_calls: nil, tool_call_id: nil),
                APIChatMessage(role: "user", content: .text("Hello"), name: nil, tool_calls: nil, tool_call_id: nil)
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

        XCTAssertEqual(prepared.instructions, "Base system\n\nExtra system")
        XCTAssertEqual(prepared.chatMessages.count, 1)
        XCTAssertEqual(prepared.chatMessages[0].content, "Hello")
        XCTAssertEqual(prepared.messageSignatures.count, 1)
        XCTAssertFalse(prepared.containsImages)
        XCTAssertNotNil(prepared.additionalContext)
        XCTAssertGreaterThan(prepared.estimatedPromptTokens, 0)
    }

    func testBuildFormatsAssistantToolCallsForQwen() {
        let toolCall = APIToolCall(
            id: "call_1",
            function: APIFunctionCall(name: "weather", arguments: "{\"city\":\"Berlin\"}")
        )
        let request = APIChatCompletionRequest(
            model: "qwen",
            messages: [
                APIChatMessage(role: "assistant", content: .text("Let me check."), name: nil, tool_calls: [toolCall], tool_call_id: nil)
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

        let prepared = PromptBuilder.build(from: request, modelId: "mlx-community/Qwen3-VL-4B-Instruct-4bit", thinkingEnabled: true)

        XCTAssertEqual(prepared.chatMessages.count, 1)
        XCTAssertTrue(prepared.chatMessages[0].content.contains("Let me check."))
        XCTAssertTrue(prepared.chatMessages[0].content.contains("<tool_call>"))
        XCTAssertNil(prepared.additionalContext)
    }

    func testBuildWrapsGemmaToolOutputsAndTracksImages() {
        let request = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(
                    role: "tool",
                    content: .parts([
                        APIContentPart(type: "text", text: "{\"ok\":true}", image_url: nil),
                        APIContentPart(type: "image_url", text: nil, image_url: APIImageURL(url: TestImageFixtures.primaryDataURI, detail: nil))
                    ]),
                    name: nil,
                    tool_calls: nil,
                    tool_call_id: "call_1"
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

        let prepared = PromptBuilder.build(from: request, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: true)

        XCTAssertTrue(prepared.chatMessages[0].content.contains("```tool_output"))
        XCTAssertTrue(prepared.containsImages)
        XCTAssertEqual(prepared.chatMessages[0].images.count, 1)
        XCTAssertEqual(prepared.imageFingerprints.count, 1)
        XCTAssertGreaterThan(prepared.estimatedBytes, prepared.chatMessages[0].content.utf8.count)
    }

    func testBuildHashesRawImageSourcesIntoStableFingerprints() {
        let firstRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(
                    role: "user",
                    content: .parts([
                        APIContentPart(type: "text", text: "Describe this.", image_url: nil),
                        APIContentPart(type: "image_url", text: nil, image_url: APIImageURL(url: TestImageFixtures.primaryDataURI, detail: nil))
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
        let secondRequest = APIChatCompletionRequest(
            model: "gemma",
            messages: [
                APIChatMessage(
                    role: "user",
                    content: .parts([
                        APIContentPart(type: "text", text: "Describe this.", image_url: nil),
                        APIContentPart(type: "image_url", text: nil, image_url: APIImageURL(url: TestImageFixtures.alternateDataURI, detail: nil))
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

        let firstPrepared = PromptBuilder.build(from: firstRequest, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: true)
        let secondPrepared = PromptBuilder.build(from: secondRequest, modelId: "mlx-community/gemma-3-4b-it-4bit", thinkingEnabled: true)

        XCTAssertEqual(firstPrepared.imageFingerprints.count, 1)
        XCTAssertEqual(secondPrepared.imageFingerprints.count, 1)
        XCTAssertNotEqual(firstPrepared.imageFingerprints, secondPrepared.imageFingerprints)
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
        let userInput = UserInput(
            prompt: .chat((instructions.isEmpty ? [] : [Chat.Message(role: .system, content: instructions)]) + chatMessages),
            images: allImages,
            videos: [],
            tools: nil,
            additionalContext: additionalContext
        )

        return PromptBuilder.PreparedPrompt(
            instructions: instructions,
            chatMessages: chatMessages,
            messageSignatures: messageSignatures,
            imageFingerprints: imageURLsFingerprintOrder(from: request),
            estimatedBytes: estimatedBytes,
            estimatedPromptTokens: (instructions.count + chatMessages.reduce(0) { $0 + $1.content.count }) * 10 / 35,
            containsImages: containsImages,
            additionalContext: additionalContext,
            userInput: userInput
        )
    }

    private func imageURLsFingerprintOrder(from request: APIChatCompletionRequest) -> [UInt64] {
        request.messages
            .filter { $0.role != "system" }
            .flatMap { $0.content?.imageURLs ?? [] }
            .reduce(into: [UInt64]()) { fingerprints, imageURL in
                var hash: UInt64 = 14_695_981_039_346_656_037
                for byte in imageURL.utf8 {
                    hash ^= UInt64(byte)
                    hash &*= 1_099_511_628_211
                }
                fingerprints.append(hash)
            }
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
}

private extension Chat.Message.Role {
    var roleLabel: String {
        switch self {
        case .assistant: "assistant"
        case .system: "system"
        case .user: "user"
        @unknown default: "unknown"
        }
    }
}