import Foundation
import MLXLMCommon

/// Defines a supported model with its metadata.
struct ModelConfig: Identifiable, Hashable {
    enum LoaderKind: Hashable {
        case llm
        case vlm
    }

    let id: String          // alias: "gemma", "gemma3n", "qwen"
    let repoId: String      // HuggingFace ID
    let displayName: String
    let contextLength: Int
    let loaderKind: LoaderKind
    let supportsImages: Bool
    let supportsTools: Bool

    /// All models supported by the app.
    static let availableModels: [ModelConfig] = [
        ModelConfig(
            id: "gemma",
            repoId: "mlx-community/gemma-3-4b-it-4bit",
            displayName: "Gemma 3 4B",
            contextLength: 128_000,
            loaderKind: .vlm,
            supportsImages: true,
            supportsTools: true
        ),
        ModelConfig(
            id: "qwen",
            repoId: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            displayName: "Qwen3 VL 4B",
            contextLength: 256_000,
            loaderKind: .vlm,
            supportsImages: true,
            supportsTools: true
        ),
        ModelConfig(
            id: "qwen3.5-9b",
            repoId: "mlx-community/Qwen3.5-9B-4bit",
            displayName: "Qwen3.5 9B",
            contextLength: 256_000,
            loaderKind: .vlm,
            supportsImages: true,
            supportsTools: true
        ),
        ModelConfig(
            id: "stheno",
            repoId: "synk/L3-8B-Stheno-v3.2-MLX",
            displayName: "Stheno L3 8B",
            contextLength: 8_192,
            loaderKind: .llm,
            supportsImages: false,
            supportsTools: false
        ),
    ]

    static let `default` = availableModels[0]

    /// Whether this model is cached locally (no download needed).
    var isLocal: Bool {
        LocalModelResolver.isAvailable(repoId: repoId)
    }

    /// Build a ModelConfiguration for mlx-swift-lm from this config.
    var modelConfiguration: ModelConfiguration {
        ModelConfiguration(id: repoId)
    }

    /// Resolve a model string (alias, full repo ID, or partial match) to a ModelConfig.
    /// Mirrors the Python server's `ModelManager.resolve_model()`.
    static func resolve(_ requested: String) -> ModelConfig? {
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
        return nil
    }
}
