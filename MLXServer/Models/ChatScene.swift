import Foundation

struct ChatScene: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    var modelId: String?
    var systemPrompt: String
    var starterPrompt: String
    var generationOverrides: GenerationSettingsOverride

    init(
        id: UUID = UUID(),
        name: String,
        modelId: String? = nil,
        systemPrompt: String = "",
        starterPrompt: String = "",
        generationOverrides: GenerationSettingsOverride = .none
    ) {
        self.id = id
        self.name = name
        self.modelId = modelId
        self.systemPrompt = systemPrompt
        self.starterPrompt = starterPrompt
        self.generationOverrides = generationOverrides
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case modelId
        case systemPrompt
        case starterPrompt
        case generationOverrides
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        modelId = try container.decodeIfPresent(String.self, forKey: .modelId)
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        starterPrompt = try container.decodeIfPresent(String.self, forKey: .starterPrompt) ?? ""
        generationOverrides = try container.decodeIfPresent(GenerationSettingsOverride.self, forKey: .generationOverrides) ?? .none
    }

    var trimmedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var displayName: String {
        trimmedName.isEmpty ? "Untitled Scene" : trimmedName
    }

    var resolvedModel: ModelConfig? {
        guard let modelId else { return nil }
        return ModelConfig.resolve(modelId)
    }

    static let empty = ChatScene(name: "New Scene")
}