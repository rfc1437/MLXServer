import Foundation
import MLXLMCommon

/// Converts OpenAI-format API messages into reusable prompt artifacts for the API server.
enum PromptBuilder {
    struct PreparedPrompt {
        let instructions: String
        let chatMessages: [Chat.Message]
        let messageSignatures: [UInt64]
        let imageFingerprints: [UInt64]
        let estimatedBytes: Int
        let estimatedPromptTokens: Int
        let containsImages: Bool
        let additionalContext: [String: any Sendable]?
        let userInput: UserInput
    }

    static func build(
        from request: APIChatCompletionRequest,
        modelId: String,
        thinkingEnabled: Bool
    ) -> PreparedPrompt {
        var instructions = ""
        for msg in request.messages where msg.role == "system" {
            let text = msg.content?.textContent ?? ""
            guard !text.isEmpty else { continue }
            if !instructions.isEmpty { instructions += "\n\n" }
            instructions += text
        }

        if let tools = request.tools, !tools.isEmpty {
            let toolPrompt = ToolPromptBuilder.buildSystemPrompt(tools: tools, modelId: modelId)
            if !instructions.isEmpty { instructions += "\n\n" }
            instructions += toolPrompt
        }

        let isQwen = modelId.lowercased().contains("qwen")
        var chatMessages: [Chat.Message] = []
        var messageSignatures: [UInt64] = []
        var imageFingerprints: [UInt64] = []
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
                text = text.isEmpty ? formattedCalls : text + "\n" + formattedCalls
            }

            let imageURLs = msg.content?.imageURLs ?? []
            var messageImages: [UserInput.Image] = []
            var messageImageBytes = 0
            for urlString in imageURLs {
                if let decoded = ImageDecoder.decode(urlString) {
                    messageImages.append(decoded.image)
                    imageFingerprints.append(imageFingerprint(urlString))
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

        var allMessages: [Chat.Message] = []
        if !instructions.isEmpty {
            allMessages.append(Chat.Message(role: .system, content: instructions))
        }
        allMessages.append(contentsOf: chatMessages)

        let allImages = chatMessages.flatMap(\ .images)
        let userInput = UserInput(
            prompt: .chat(allMessages),
            images: allImages,
            videos: [],
            tools: nil,
            additionalContext: additionalContext
        )

        let estimatedPromptTokens = (instructions.count + chatMessages.reduce(0) { $0 + $1.content.count }) * 10 / 35

        return PreparedPrompt(
            instructions: instructions,
            chatMessages: chatMessages,
            messageSignatures: messageSignatures,
            imageFingerprints: imageFingerprints,
            estimatedBytes: estimatedBytes,
            estimatedPromptTokens: estimatedPromptTokens,
            containsImages: containsImages,
            additionalContext: additionalContext,
            userInput: userInput
        )
    }

    private static func imageFingerprint(_ source: String) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in source.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return hash
    }

    private static func messageSignature(role: Chat.Message.Role, content: String, imageURLs: [String]) -> UInt64 {
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