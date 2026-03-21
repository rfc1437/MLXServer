import Foundation
import XCTest
@testable import MLX_Server

final class LocalModelResolverTests: XCTestCase {
    func testDiscoverModelsInfersTextOnlyMetadataAndDirectorySize() throws {
        let base = try makeTempModelsRoot()
        let repoDirectory = try makeRepoDirectory(base: base, owner: "example", repo: "text-only")
        let configURL = repoDirectory.appendingPathComponent("config.json")
        let modelURL = repoDirectory.appendingPathComponent("model.safetensors")
        let tokenizerURL = repoDirectory.appendingPathComponent("tokenizer.json")

        try writeJSON(
            [
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 32768,
            ],
            to: configURL
        )
        try Data(repeating: 0x11, count: 64).write(to: modelURL)
        try Data(repeating: 0x22, count: 19).write(to: tokenizerURL)

        let expectedSize = Int64(
            try Data(contentsOf: configURL).count
            + Data(contentsOf: modelURL).count
            + Data(contentsOf: tokenizerURL).count
        )

        let discovered = LocalModelResolver.discoverModels(in: base)
        let model = try XCTUnwrap(discovered.first)

        XCTAssertEqual(model.repoId, "example/text-only")
        XCTAssertEqual(model.contextLength, 32768)
        XCTAssertFalse(model.supportsImages)
        XCTAssertEqual(model.loaderKinds, [.llm, .vlm])
        XCTAssertEqual(model.sizeBytes, expectedSize)
    }

    func testDiscoverModelsInfersVisionMetadataFromProcessorFiles() throws {
        let base = try makeTempModelsRoot()
        let repoDirectory = try makeRepoDirectory(base: base, owner: "example", repo: "vision-model")
        try writeJSON(
            [
                "text_config": ["max_position_embeddings": 262144],
                "vision_config": ["hidden_size": 768],
            ],
            to: repoDirectory.appendingPathComponent("config.json")
        )
        try writeJSON(["processor_class": "Qwen3VLProcessor"], to: repoDirectory.appendingPathComponent("tokenizer_config.json"))
        try Data(repeating: 0x33, count: 12).write(to: repoDirectory.appendingPathComponent("processor_config.json"))
        try Data(repeating: 0x44, count: 8).write(to: repoDirectory.appendingPathComponent("model.safetensors.index.json"))

        let discovered = LocalModelResolver.discoverModels(in: base)
        let model = try XCTUnwrap(discovered.first)

        XCTAssertEqual(model.repoId, "example/vision-model")
        XCTAssertEqual(model.contextLength, 262144)
        XCTAssertTrue(model.supportsImages)
        XCTAssertEqual(model.loaderKinds, [.vlm, .llm])
    }

    func testMergedCatalogKeepsCuratedModelsAndAddsCustomLocalModels() {
        let localModels = [
            LocalModelResolver.LocalModelInfo(
                repoId: "mlx-community/gemma-3-4b-it-4bit",
                directory: URL(fileURLWithPath: "/tmp/gemma"),
                sizeBytes: 1024,
                contextLength: 128000,
                loaderKinds: [.vlm, .llm],
                supportsImages: true
            ),
            LocalModelResolver.LocalModelInfo(
                repoId: "custom-org/custom-model",
                directory: URL(fileURLWithPath: "/tmp/custom"),
                sizeBytes: 2048,
                contextLength: 65536,
                loaderKinds: [.llm, .vlm],
                supportsImages: false
            ),
        ]

        let merged = ModelConfig.mergedModels(localModels: localModels)
        let gemma = merged.first(where: { $0.id == "gemma" })
        let custom = merged.first(where: { $0.repoId == "custom-org/custom-model" })

        XCTAssertEqual(gemma?.localSizeBytes, 1024)
        XCTAssertEqual(custom?.id, "custom-org/custom-model")
        XCTAssertEqual(custom?.contextLength, 65536)
        XCTAssertFalse(custom?.isCurated ?? true)
    }

    func testResolveUnknownRepoIdCreatesRemoteCustomConfig() throws {
        let config = try XCTUnwrap(ModelConfig.resolve("custom-owner/custom-repo"))

        XCTAssertEqual(config.id, "custom-owner/custom-repo")
        XCTAssertEqual(config.repoId, "custom-owner/custom-repo")
        XCTAssertFalse(config.isCurated)
    }

    func testMergedCatalogAppliesSavedMetadataOverride() {
        let repoId = "custom-org/override-model"
        Preferences.setModelMetadataOverride(
            ModelMetadataOverride(
                contextLength: 123456,
                primaryLoaderKind: .vlm,
                supportsImages: true,
                supportsTools: true
            ),
            forRepoId: repoId
        )
        defer {
            Preferences.removeModelMetadataOverride(forRepoId: repoId)
        }

        let localModels = [
            LocalModelResolver.LocalModelInfo(
                repoId: repoId,
                directory: URL(fileURLWithPath: "/tmp/custom-override"),
                sizeBytes: 2048,
                contextLength: 65536,
                loaderKinds: [.llm, .vlm],
                supportsImages: false
            ),
        ]

        let merged = ModelConfig.mergedModels(localModels: localModels)
        let overridden = merged.first(where: { $0.repoId == repoId })

        XCTAssertEqual(overridden?.contextLength, 123456)
        XCTAssertEqual(overridden?.primaryLoaderKind, .vlm)
        XCTAssertTrue(overridden?.supportsImages ?? false)
        XCTAssertTrue(overridden?.supportsTools ?? false)
    }

    func testResolveUnknownRepoIdUsesSavedMetadataOverride() throws {
        let repoId = "custom-owner/custom-repo-with-override"
        Preferences.setModelMetadataOverride(
            ModelMetadataOverride(
                contextLength: 8192,
                primaryLoaderKind: .llm,
                supportsImages: false,
                supportsTools: true
            ),
            forRepoId: repoId
        )
        defer {
            Preferences.removeModelMetadataOverride(forRepoId: repoId)
        }

        let config = try XCTUnwrap(ModelConfig.resolve(repoId))

        XCTAssertEqual(config.contextLength, 8192)
        XCTAssertEqual(config.primaryLoaderKind, .llm)
        XCTAssertFalse(config.supportsImages)
        XCTAssertTrue(config.supportsTools)
    }

    private func makeTempModelsRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }

    private func makeRepoDirectory(base: URL, owner: String, repo: String) throws -> URL {
        let directory = base
            .appendingPathComponent(owner, isDirectory: true)
            .appendingPathComponent(repo, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func writeJSON(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}