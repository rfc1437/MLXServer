import Foundation

struct GenerationSettings: Codable, Hashable, Sendable {
    var temperature: Double
    var topP: Double
    var topK: Int
    var minP: Double
    var maxTokens: Int
    var repetitionPenalty: Double?
    var presencePenalty: Double?
    var frequencyPenalty: Double?
    var thinkingEnabled: Bool

    init(
        temperature: Double = 0.7,
        topP: Double = 1.0,
        topK: Int = 0,
        minP: Double = 0.0,
        maxTokens: Int = 4096,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        thinkingEnabled: Bool = true
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.thinkingEnabled = thinkingEnabled
    }

    func normalized() -> GenerationSettings {
        GenerationSettings(
            temperature: max(0, temperature),
            topP: min(max(topP, 0), 1),
            topK: max(0, topK),
            minP: min(max(minP, 0), 1),
            maxTokens: max(1, maxTokens),
            repetitionPenalty: Self.normalizePositive(repetitionPenalty),
            presencePenalty: Self.normalizeSignedPenalty(presencePenalty),
            frequencyPenalty: Self.normalizeSignedPenalty(frequencyPenalty),
            thinkingEnabled: thinkingEnabled
        )
    }

    func applying(_ overrides: GenerationSettingsOverride) -> GenerationSettings {
        GenerationSettings(
            temperature: overrides.temperature ?? temperature,
            topP: overrides.topP ?? topP,
            topK: overrides.topK ?? topK,
            minP: overrides.minP ?? minP,
            maxTokens: overrides.maxTokens ?? maxTokens,
            repetitionPenalty: overrides.repetitionPenalty ?? repetitionPenalty,
            presencePenalty: overrides.presencePenalty ?? presencePenalty,
            frequencyPenalty: overrides.frequencyPenalty ?? frequencyPenalty,
            thinkingEnabled: overrides.thinkingEnabled ?? thinkingEnabled
        )
        .normalized()
    }

    static func modelDefault(for modelId: String, legacyThinkingEnabled: Bool = true) -> GenerationSettings {
        let fallback = ModelConfig.resolve(modelId)?.defaultGenerationSettings ?? .generalDefault
        var resolved = fallback
        if !legacyThinkingEnabled {
            resolved.thinkingEnabled = false
        }
        return resolved.normalized()
    }

    static let generalDefault = GenerationSettings()

    static let technicalDefault = GenerationSettings(
        temperature: 0.35,
        topP: 0.9,
        topK: 40,
        minP: 0.0,
        maxTokens: 4096,
        repetitionPenalty: 1.05,
        presencePenalty: nil,
        frequencyPenalty: nil,
        thinkingEnabled: true
    )

    static let roleplayDefault = GenerationSettings(
        temperature: 0.85,
        topP: 0.95,
        topK: 60,
        minP: 0.0,
        maxTokens: 4096,
        repetitionPenalty: 1.02,
        presencePenalty: nil,
        frequencyPenalty: nil,
        thinkingEnabled: false
    )

    private static func normalizePositive(_ value: Double?) -> Double? {
        guard let value else { return nil }
        return value > 0 ? value : nil
    }

    private static func normalizeSignedPenalty(_ value: Double?) -> Double? {
        guard let value else { return nil }
        return min(max(value, -2), 2)
    }
}

struct GenerationSettingsOverride: Codable, Hashable, Sendable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var minP: Double?
    var maxTokens: Int?
    var repetitionPenalty: Double?
    var presencePenalty: Double?
    var frequencyPenalty: Double?
    var thinkingEnabled: Bool?

    init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        maxTokens: Int? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        thinkingEnabled: Bool? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.thinkingEnabled = thinkingEnabled
    }

    static let none = GenerationSettingsOverride()

    var hasOverrides: Bool {
        temperature != nil
            || topP != nil
            || topK != nil
            || minP != nil
            || maxTokens != nil
            || repetitionPenalty != nil
            || presencePenalty != nil
            || frequencyPenalty != nil
            || thinkingEnabled != nil
    }
}