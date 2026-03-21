import Foundation
import MLXLMCommon

struct ModelMetadataOverride: Codable, Hashable, Sendable {
    var contextLength: Int
    var primaryLoaderKind: ModelConfig.LoaderKind
    var supportsImages: Bool
    var supportsTools: Bool

    func normalized() -> ModelMetadataOverride {
        ModelMetadataOverride(
            contextLength: max(0, contextLength),
            primaryLoaderKind: primaryLoaderKind,
            supportsImages: supportsImages,
            supportsTools: supportsTools
        )
    }
}

/// Defines a supported model with its metadata.
struct ModelConfig: Identifiable, Hashable {
    enum LoaderKind: String, CaseIterable, Codable, Hashable, Sendable {
        case llm
        case vlm

        var displayName: String {
            switch self {
            case .llm:
                return "Text"
            case .vlm:
                return "Vision"
            }
        }
    }

    let id: String          // alias: "gemma", "gemma3n", "qwen"
    let repoId: String      // HuggingFace ID
    let displayName: String
    let contextLength: Int
    let loaderKinds: [LoaderKind]
    let supportsImages: Bool
    let supportsTools: Bool
    let defaultGenerationSettings: GenerationSettings
    let isCurated: Bool
    let localSizeBytes: Int64?

    init(
        id: String,
        repoId: String,
        displayName: String,
        contextLength: Int,
        loaderKinds: [LoaderKind],
        supportsImages: Bool,
        supportsTools: Bool,
        defaultGenerationSettings: GenerationSettings,
        isCurated: Bool = true,
        localSizeBytes: Int64? = nil
    ) {
        self.id = id
        self.repoId = repoId
        self.displayName = displayName
        self.contextLength = contextLength
        self.loaderKinds = loaderKinds
        self.supportsImages = supportsImages
        self.supportsTools = supportsTools
        self.defaultGenerationSettings = defaultGenerationSettings
        self.isCurated = isCurated
        self.localSizeBytes = localSizeBytes
    }

    /// Curated models supported and tuned by the app.
    static let curatedModels: [ModelConfig] = [
        ModelConfig(
            id: "gemma",
            repoId: "mlx-community/gemma-3-4b-it-4bit",
            displayName: "Gemma 3 4B",
            contextLength: 128_000,
            loaderKinds: [.vlm],
            supportsImages: true,
            supportsTools: true,
            defaultGenerationSettings: .technicalDefault
        ),
        ModelConfig(
            id: "qwen",
            repoId: "mlx-community/Qwen3.5-4B-MLX-4bit",
            displayName: "Qwen3.5 4B",
            contextLength: 256_000,
            loaderKinds: [.vlm],
            supportsImages: true,
            supportsTools: true,
            defaultGenerationSettings: .technicalDefault
        ),
        ModelConfig(
            id: "qwen3.5-0.8b",
            repoId: "mlx-community/Qwen3.5-0.8B-4bit",
            displayName: "Qwen3.5 0.8B",
            contextLength: 256_000,
            loaderKinds: [.vlm],
            supportsImages: true,
            supportsTools: true,
            defaultGenerationSettings: .technicalDefault
        ),
        ModelConfig(
            id: "qwen3.5-9b",
            repoId: "mlx-community/Qwen3.5-9B-4bit",
            displayName: "Qwen3.5 9B",
            contextLength: 256_000,
            loaderKinds: [.vlm],
            supportsImages: true,
            supportsTools: true,
            defaultGenerationSettings: .technicalDefault
        ),
        ModelConfig(
            id: "stheno",
            repoId: "synk/L3-8B-Stheno-v3.2-MLX",
            displayName: "Stheno L3 8B",
            contextLength: 8_192,
            loaderKinds: [.llm],
            supportsImages: false,
            supportsTools: false,
            defaultGenerationSettings: .roleplayDefault
        ),
        ModelConfig(
            id: "violet-lotus",
            repoId: "hobaratio/MN-Violet-Lotus-12B-mlx-4Bit",
            displayName: "Violet Lotus 12B",
            contextLength: 32_768,
            loaderKinds: [.llm],
            supportsImages: false,
            supportsTools: false,
            defaultGenerationSettings: .roleplayDefault
        ),
    ]

    static var availableModels: [ModelConfig] {
        mergedModels(localModels: LocalModelResolver.discoveredLocalModels())
    }

    static let `default` = curatedModels[0]

    /// Whether this model is cached locally (no download needed).
    var isLocal: Bool {
        localSizeBytes != nil || LocalModelResolver.isAvailable(repoId: repoId)
    }

    var primaryLoaderKind: LoaderKind {
        loaderKinds.first ?? .llm
    }

    var metadataOverrideValue: ModelMetadataOverride {
        ModelMetadataOverride(
            contextLength: contextLength,
            primaryLoaderKind: primaryLoaderKind,
            supportsImages: supportsImages,
            supportsTools: supportsTools
        )
    }

    /// Build a ModelConfiguration for mlx-swift-lm from this config.
    var modelConfiguration: ModelConfiguration {
        ModelConfiguration(id: repoId)
    }

    /// Resolve a model string (alias, full repo ID, or partial match) to a ModelConfig.
    /// Mirrors the Python server's `ModelManager.resolve_model()`.
    static func resolve(_ requested: String) -> ModelConfig? {
        let requested = requested.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requested.isEmpty else { return nil }

        // Exact alias match
        if let config = availableModels.first(where: { $0.id == requested }) {
            return config
        }
        // Exact repo ID match
        if let config = availableModels.first(where: { $0.repoId == requested }) {
            return config
        }
        // Partial match (e.g. "gemma-3-4b-it" matches the gemma entry)
        if let config = availableModels.first(where: { requested.contains($0.id) || $0.repoId.contains(requested) || requested.contains($0.repoId) }) {
            return config
        }
        if requested.contains("/") {
            return remoteCustom(repoId: requested)
        }
        return nil
    }

    static func mergedModels(
        localModels: [LocalModelResolver.LocalModelInfo],
        applyingOverrides: Bool = true
    ) -> [ModelConfig] {
        let localByRepo = Dictionary(uniqueKeysWithValues: localModels.map { ($0.repoId, $0) })
        let curatedRepoIds = Set(curatedModels.map(\.repoId))

        let curated = curatedModels.map { config in
            if let local = localByRepo[config.repoId] {
                return applyingOverrides ? applyMetadataOverrideIfNeeded(to: config.withLocalSize(local.sizeBytes)) : config.withLocalSize(local.sizeBytes)
            }
            return applyingOverrides ? applyMetadataOverrideIfNeeded(to: config) : config
        }

        let discoveredCustom = localModels
            .filter { !curatedRepoIds.contains($0.repoId) }
            .map(customLocal)
            .sorted { lhs, rhs in
                lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
            }

        return curated + discoveredCustom
    }

    static func baselineModel(
        forRepoId repoId: String,
        localModels: [LocalModelResolver.LocalModelInfo]
    ) -> ModelConfig? {
        mergedModels(localModels: localModels, applyingOverrides: false)
            .first(where: { $0.repoId == repoId || $0.id == repoId })
            ?? (repoId.contains("/") ? remoteCustom(repoId: repoId) : nil)
    }

    static func remoteCustom(repoId: String) -> ModelConfig {
        let supportsImages = inferredVisionSupport(repoId: repoId)
        return applyMetadataOverrideIfNeeded(to: ModelConfig(
            id: repoId,
            repoId: repoId,
            displayName: displayName(for: repoId),
            contextLength: 0,
            loaderKinds: supportsImages ? [.vlm, .llm] : [.llm, .vlm],
            supportsImages: supportsImages,
            supportsTools: inferredToolSupport(repoId: repoId),
            defaultGenerationSettings: .generalDefault,
            isCurated: false
        ))
    }

    static func displayName(for repoId: String) -> String {
        let raw = repoId.split(separator: "/").last.map(String.init) ?? repoId
        return raw
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }

    private static func customLocal(_ local: LocalModelResolver.LocalModelInfo) -> ModelConfig {
        applyMetadataOverrideIfNeeded(to: ModelConfig(
            id: local.repoId,
            repoId: local.repoId,
            displayName: displayName(for: local.repoId),
            contextLength: local.contextLength,
            loaderKinds: local.loaderKinds,
            supportsImages: local.supportsImages,
            supportsTools: inferredToolSupport(repoId: local.repoId),
            defaultGenerationSettings: .generalDefault,
            isCurated: false,
            localSizeBytes: local.sizeBytes
        ))
    }

    private static func inferredToolSupport(repoId: String) -> Bool {
        let normalized = repoId.lowercased()
        return normalized.contains("qwen") || normalized.contains("gemma")
    }

    private static func inferredVisionSupport(repoId: String) -> Bool {
        let normalized = repoId.lowercased()
        return normalized.contains("vision") || normalized.contains("vl") || normalized.contains("gemma-3") || normalized.contains("qwen")
    }

    private func withLocalSize(_ sizeBytes: Int64) -> ModelConfig {
        ModelConfig(
            id: id,
            repoId: repoId,
            displayName: displayName,
            contextLength: contextLength,
            loaderKinds: loaderKinds,
            supportsImages: supportsImages,
            supportsTools: supportsTools,
            defaultGenerationSettings: defaultGenerationSettings,
            isCurated: isCurated,
            localSizeBytes: sizeBytes
        )
    }

    private func applyingMetadataOverride(_ override: ModelMetadataOverride) -> ModelConfig {
        let normalized = override.normalized()
        let reorderedLoaderKinds = [normalized.primaryLoaderKind] + LoaderKind.allCases.filter { $0 != normalized.primaryLoaderKind }

        return ModelConfig(
            id: id,
            repoId: repoId,
            displayName: displayName,
            contextLength: normalized.contextLength,
            loaderKinds: reorderedLoaderKinds,
            supportsImages: normalized.supportsImages,
            supportsTools: normalized.supportsTools,
            defaultGenerationSettings: defaultGenerationSettings,
            isCurated: isCurated,
            localSizeBytes: localSizeBytes
        )
    }

    private static func applyMetadataOverrideIfNeeded(to config: ModelConfig) -> ModelConfig {
        guard let override = Preferences.modelMetadataOverride(forRepoId: config.repoId) else {
            return config
        }
        return config.applyingMetadataOverride(override)
    }
}
