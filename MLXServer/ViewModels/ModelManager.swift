import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// Manages model loading, switching, and generation.
@Observable
@MainActor
final class ModelManager {

    /// HubApi with blob cache disabled to avoid storing every model twice.
    /// swift-huggingface defaults to caching in both huggingface/hub/ (snapshots)
    /// AND models/ (content-addressed blobs). We only need the snapshots.
    /// Must use the same downloadBase as defaultHubApi (.cachesDirectory) so
    /// LocalModelResolver can find downloaded models.
    private static let hub: HubApi = {
        let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        return HubApi(downloadBase: cachesDir, cache: nil)
    }()

    var currentModel: ModelConfig?
    var availableModels: [ModelConfig]
    private(set) var discoveredLocalModels: [LocalModelResolver.LocalModelInfo] = []
    var modelContainer: ModelContainer?
    var isLoading = false
    var downloadProgress: Double = 0
    var loadingModelName: String = ""
    var errorMessage: String?

    // Download-specific state for the modal
    var isDownloading = false
    var downloadFilesTotal: Int64 = 0
    var downloadFilesCompleted: Int64 = 0
    var downloadSpeed: Double = 0 // bytes/sec

    private var idleTimer: Timer?
    private(set) var lastUsed: Date?
    private var latestLoadRequestID = UUID()

    init() {
        availableModels = []
        refreshAvailableModels()
    }

    var curatedModels: [ModelConfig] {
        availableModels.filter(\.isCurated)
    }

    var localModelsOnDisk: [ModelConfig] {
        availableModels
            .filter(\.isLocal)
            .sorted {
                $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
            }
    }

    func refreshAvailableModels() {
        discoveredLocalModels = LocalModelResolver.discoveredLocalModels()
        availableModels = ModelConfig.mergedModels(localModels: discoveredLocalModels)

        if let currentModel {
            self.currentModel = availableModels.first(where: { $0.repoId == currentModel.repoId }) ?? currentModel
        }
    }

    func discoveredLocalModelInfo(repoId: String) -> LocalModelResolver.LocalModelInfo? {
        discoveredLocalModels.first(where: { $0.repoId == repoId })
    }

    func baselineModel(repoId: String) -> ModelConfig? {
        ModelConfig.baselineModel(forRepoId: repoId, localModels: discoveredLocalModels)
    }

    func saveMetadataOverride(_ override: ModelMetadataOverride, for config: ModelConfig) {
        Preferences.setModelMetadataOverride(override, forRepoId: config.repoId)
        refreshAvailableModels()
    }

    func clearMetadataOverride(for config: ModelConfig) {
        Preferences.removeModelMetadataOverride(forRepoId: config.repoId)
        refreshAvailableModels()
    }

    private func clearLoadedState() {
        idleTimer?.invalidate()
        idleTimer = nil
        lastUsed = nil
        modelContainer = nil
        currentModel = nil
        isLoading = false
        isDownloading = false
        downloadProgress = 0
        loadingModelName = ""
        downloadFilesTotal = 0
        downloadFilesCompleted = 0
        downloadSpeed = 0
    }

    /// Load a model, unloading the current one first.
    /// Prefers the local snapshot from ~/.cache/huggingface/hub/ (shared with the Python server).
    /// Only downloads if the model isn't cached locally.
    func loadModel(_ config: ModelConfig) async {
        refreshAvailableModels()
        let effectiveConfig = availableModels.first(where: { $0.repoId == config.repoId }) ?? config

        if currentModel?.repoId == effectiveConfig.repoId && modelContainer != nil {
            currentModel = effectiveConfig
            return // already loaded
        }

        let requestID = UUID()
        latestLoadRequestID = requestID
        clearLoadedState()
        MLX.GPU.clearCache()
        isLoading = true
        downloadProgress = 0
        loadingModelName = effectiveConfig.displayName
        errorMessage = nil

        let needsDownload = !effectiveConfig.isLocal
        if needsDownload {
            isDownloading = true
            downloadFilesTotal = 0
            downloadFilesCompleted = 0
            downloadSpeed = 0
        }

        do {
            let progressHandler: @Sendable (Progress) -> Void = { progress in
                Task { @MainActor in
                    self.downloadProgress = progress.fractionCompleted
                    if self.isDownloading {
                        self.downloadFilesTotal = progress.totalUnitCount
                        self.downloadFilesCompleted = progress.completedUnitCount
                        if let speed = progress.userInfo[.throughputKey] as? Double {
                            self.downloadSpeed = speed
                        }
                    }
                }
            }

            let configuration: ModelConfiguration
            if let localDir = LocalModelResolver.resolve(repoId: effectiveConfig.repoId) {
                configuration = ModelConfiguration(directory: localDir)
            } else {
                configuration = effectiveConfig.modelConfiguration
            }

            let container = try await Self.loadContainer(
                for: effectiveConfig,
                configuration: configuration,
                progressHandler: progressHandler
            )

            guard latestLoadRequestID == requestID else { return }
            refreshAvailableModels()
            self.isDownloading = false
            self.modelContainer = container
            self.currentModel = self.availableModels.first(where: { $0.repoId == effectiveConfig.repoId }) ?? effectiveConfig
            touchActivity()
        } catch {
            guard latestLoadRequestID == requestID else { return }
            self.isDownloading = false
            self.errorMessage = "Failed to load model: \(error.localizedDescription)"
        }

        guard latestLoadRequestID == requestID else { return }
        isLoading = false
    }

    /// Delete local cache and re-download a model.
    func redownloadModel(_ config: ModelConfig) async {
        unloadModel()
        LocalModelResolver.deleteLocal(repoId: config.repoId)
        await loadModel(config)
    }

    func addModel(repoId: String) async {
        let repoId = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoId.isEmpty else {
            errorMessage = "Enter a HuggingFace model ID."
            return
        }

        let config = ModelConfig.resolve(repoId) ?? ModelConfig.remoteCustom(repoId: repoId)
        await loadModel(config)
    }

    func deleteModel(_ config: ModelConfig) {
        if currentModel?.repoId == config.repoId {
            unloadModel()
        }
        _ = LocalModelResolver.deleteLocal(repoId: config.repoId)
        refreshAvailableModels()
    }

    /// Unload the current model and free GPU memory.
    func unloadModel() {
        latestLoadRequestID = UUID()
        clearLoadedState()
        MLX.GPU.clearCache()
    }

    /// Record model activity and reset the idle unload timer.
    func touchActivity() {
        lastUsed = Date()
        idleTimer?.invalidate()
        let minutes = Preferences.idleUnloadMinutes
        guard minutes > 0 else { return }
        idleTimer = Timer.scheduledTimer(withTimeInterval: TimeInterval(minutes * 60), repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.modelContainer != nil else { return }
                print("[ModelManager] Idle for \(minutes) min — unloading model")
                self.unloadModel()
            }
        }
    }

    /// Whether a model is ready for generation.
    var isReady: Bool {
        modelContainer != nil && !isLoading
    }

    private static func loadContainer(
        for config: ModelConfig,
        configuration: ModelConfiguration,
        progressHandler: @escaping @Sendable (Progress) -> Void
    ) async throws -> ModelContainer {
        var lastError: Error?

        for loaderKind in config.loaderKinds {
            do {
                switch loaderKind {
                case .llm:
                    return try await LLMModelFactory.shared.loadContainer(
                        hub: Self.hub,
                        configuration: configuration,
                        progressHandler: progressHandler
                    )
                case .vlm:
                    return try await VLMModelFactory.shared.loadContainer(
                        hub: Self.hub,
                        configuration: configuration,
                        progressHandler: progressHandler
                    )
                }
            } catch {
                lastError = error
            }
        }

        throw lastError ?? NSError(domain: "ModelManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unsupported model configuration"])
    }
}
