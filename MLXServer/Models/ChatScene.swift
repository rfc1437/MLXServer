import Foundation

struct ChatScene: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    var modelId: String?
    var systemPrompt: String
    var starterPrompt: String

    init(
        id: UUID = UUID(),
        name: String,
        modelId: String? = nil,
        systemPrompt: String = "",
        starterPrompt: String = ""
    ) {
        self.id = id
        self.name = name
        self.modelId = modelId
        self.systemPrompt = systemPrompt
        self.starterPrompt = starterPrompt
    }

    var trimmedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var displayName: String {
        trimmedName.isEmpty ? "Untitled Scene" : trimmedName
    }

    var resolvedModel: ModelConfig? {
        guard let modelId else { return nil }
        return ModelConfig.availableModels.first(where: { $0.id == modelId })
    }

    static let empty = ChatScene(name: "New Scene")
}