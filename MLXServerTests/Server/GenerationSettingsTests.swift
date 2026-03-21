import XCTest
@testable import MLX_Server

final class GenerationSettingsTests: XCTestCase {
    func testSceneOverridesApplyWithoutDiscardingModelDefaults() {
        let base = GenerationSettings(
            temperature: 0.2,
            topP: 0.9,
            topK: 12,
            minP: 0.05,
            maxTokens: 2048,
            repetitionPenalty: 1.08,
            presencePenalty: 0.3,
            frequencyPenalty: 0.1,
            thinkingEnabled: true
        )

        let overrides = GenerationSettingsOverride(
            temperature: 0.8,
            repetitionPenalty: 1.2,
            thinkingEnabled: false
        )

        let resolved = base.applying(overrides)

        XCTAssertEqual(resolved.temperature, 0.8)
        XCTAssertEqual(resolved.repetitionPenalty, 1.2)
        XCTAssertEqual(resolved.topP, 0.9)
        XCTAssertEqual(resolved.topK, 12)
        XCTAssertEqual(resolved.maxTokens, 2048)
        XCTAssertEqual(resolved.presencePenalty, 0.3)
        XCTAssertFalse(resolved.thinkingEnabled)
    }

    func testPreferencesStoreGenerationDefaultsPerModel() {
        let gemmaId = "gemma"
        let qwenId = "qwen3.5-0.8b"
        let originalGemma = Preferences.generationSettings(forModelId: gemmaId)
        let originalQwen = Preferences.generationSettings(forModelId: qwenId)

        defer {
            Preferences.setGenerationSettings(originalGemma, forModelId: gemmaId)
            Preferences.setGenerationSettings(originalQwen, forModelId: qwenId)
        }

        Preferences.setGenerationSettings(
            GenerationSettings(temperature: 0.15, topP: 0.85, maxTokens: 1024, repetitionPenalty: 1.1, thinkingEnabled: false),
            forModelId: gemmaId
        )
        Preferences.setGenerationSettings(
            GenerationSettings(temperature: 0.95, topP: 1.0, maxTokens: 8192, repetitionPenalty: nil, thinkingEnabled: true),
            forModelId: qwenId
        )

        let gemma = Preferences.generationSettings(forModelId: gemmaId)
        let qwen = Preferences.generationSettings(forModelId: qwenId)

        XCTAssertEqual(gemma.temperature, 0.15)
        XCTAssertEqual(gemma.topP, 0.85)
        XCTAssertEqual(gemma.maxTokens, 1024)
        XCTAssertEqual(gemma.repetitionPenalty, 1.1)
        XCTAssertFalse(gemma.thinkingEnabled)

        XCTAssertEqual(qwen.temperature, 0.95)
        XCTAssertEqual(qwen.maxTokens, 8192)
        XCTAssertNil(qwen.repetitionPenalty)
        XCTAssertTrue(qwen.thinkingEnabled)
    }

    func testModelFallbackDefaultsComeFromModelDefinitions() {
        let gemma = GenerationSettings.modelDefault(for: "gemma")
        let qwen = GenerationSettings.modelDefault(for: "qwen")
        let stheno = GenerationSettings.modelDefault(for: "stheno")

        XCTAssertEqual(gemma, .technicalDefault)
        XCTAssertEqual(qwen, .technicalDefault)
        XCTAssertEqual(stheno, .roleplayDefault)
        XCTAssertNotEqual(gemma, stheno)
    }
}