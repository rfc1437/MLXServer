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
        if currentModel?.id == config.id && modelContainer != nil {
            return // already loaded
        }

        let requestID = UUID()
        latestLoadRequestID = requestID
        clearLoadedState()
        MLX.GPU.clearCache()
        isLoading = true
        downloadProgress = 0
        loadingModelName = config.displayName
        errorMessage = nil

        let needsDownload = !config.isLocal
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
            if let localDir = LocalModelResolver.resolve(repoId: config.repoId) {
                configuration = ModelConfiguration(directory: localDir)
            } else {
                configuration = config.modelConfiguration
            }

            let container: ModelContainer
            switch config.loaderKind {
            case .llm:
                container = try await LLMModelFactory.shared.loadContainer(
                    hub: Self.hub,
                    configuration: configuration,
                    progressHandler: progressHandler
                )
            case .vlm:
                container = try await VLMModelFactory.shared.loadContainer(
                    hub: Self.hub,
                    configuration: configuration,
                    progressHandler: progressHandler
                )
            }

            guard latestLoadRequestID == requestID else { return }
            self.isDownloading = false
            self.modelContainer = container
            self.currentModel = config
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
}
