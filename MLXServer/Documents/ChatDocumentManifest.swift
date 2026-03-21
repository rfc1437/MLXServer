import Foundation

struct ChatDocumentManifest: Codable {
    var schemaVersion: Int
    var documentId: UUID
    var createdAt: Date
    var updatedAt: Date
    var appVersion: String
    var model: StoredModelInfo?
    var settings: StoredChatSettings
    var messages: [StoredChatMessage]
    var uiState: StoredChatUIState

    static let currentSchemaVersion = 2

    struct StoredModelInfo: Codable, Hashable {
        var id: String
        var displayName: String
        var repoId: String
    }

    struct StoredChatSettings: Codable, Hashable {
        var systemPrompt: String
        var thinkingEnabled: Bool
        var temperature: Double
        var topP: Double
        var topK: Int
        var minP: Double
        var maxTokens: Int
        var repetitionPenalty: Double?
        var presencePenalty: Double?
        var frequencyPenalty: Double?

        init(systemPrompt: String, generationSettings: GenerationSettings) {
            self.systemPrompt = systemPrompt
            self.thinkingEnabled = generationSettings.thinkingEnabled
            self.temperature = generationSettings.temperature
            self.topP = generationSettings.topP
            self.topK = generationSettings.topK
            self.minP = generationSettings.minP
            self.maxTokens = generationSettings.maxTokens
            self.repetitionPenalty = generationSettings.repetitionPenalty
            self.presencePenalty = generationSettings.presencePenalty
            self.frequencyPenalty = generationSettings.frequencyPenalty
        }

        var generationSettings: GenerationSettings {
            GenerationSettings(
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                presencePenalty: presencePenalty,
                frequencyPenalty: frequencyPenalty,
                thinkingEnabled: thinkingEnabled
            ).normalized()
        }

        private enum CodingKeys: String, CodingKey {
            case systemPrompt
            case thinkingEnabled
            case temperature
            case topP
            case topK
            case minP
            case maxTokens
            case repetitionPenalty
            case presencePenalty
            case frequencyPenalty
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let fallback = GenerationSettings()

            systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
            thinkingEnabled = try container.decodeIfPresent(Bool.self, forKey: .thinkingEnabled) ?? fallback.thinkingEnabled
            temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? fallback.temperature
            topP = try container.decodeIfPresent(Double.self, forKey: .topP) ?? fallback.topP
            topK = try container.decodeIfPresent(Int.self, forKey: .topK) ?? fallback.topK
            minP = try container.decodeIfPresent(Double.self, forKey: .minP) ?? fallback.minP
            maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? fallback.maxTokens
            repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
            presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty)
            frequencyPenalty = try container.decodeIfPresent(Double.self, forKey: .frequencyPenalty)
        }
    }

    struct StoredChatUIState: Codable, Hashable {
        var draftInput: String
        var scrollAnchorMessageId: UUID?
    }

    struct StoredChatMessage: Codable, Hashable, Identifiable {
        enum Role: String, Codable {
            case system
            case user
            case assistant
        }

        enum StreamingState: String, Codable {
            case streaming
            case completed
        }

        var id: UUID
        var role: Role
        var createdAt: Date
        var content: String
        var rawContent: String
        var thinkingContent: String
        var streamingState: StreamingState
        var attachments: [StoredAttachment]
    }

    struct StoredAttachment: Codable, Hashable, Identifiable {
        var id: UUID
        var type: String
        var relativePath: String
        var mimeType: String
        var pixelWidth: Int?
        var pixelHeight: Int?
        var sha256: String
    }
}

struct ChatDocumentSnapshot: Codable, Hashable {
    var documentId: UUID
    var createdAt: Date
    var model: ChatDocumentManifest.StoredModelInfo?
    var settings: ChatDocumentManifest.StoredChatSettings
    var messages: [ChatDocumentManifest.StoredChatMessage]
    var uiState: ChatDocumentManifest.StoredChatUIState
}
