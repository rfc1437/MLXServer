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

    static let currentSchemaVersion = 1

    struct StoredModelInfo: Codable, Hashable {
        var id: String
        var displayName: String
        var repoId: String
    }

    struct StoredChatSettings: Codable, Hashable {
        var systemPrompt: String
        var thinkingEnabled: Bool
        var temperature: Double
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
