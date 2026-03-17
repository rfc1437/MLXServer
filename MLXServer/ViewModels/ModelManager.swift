import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// Manages model loading, switching, and generation.
@Observable
@MainActor
final class ModelManager {
    var currentModel: ModelConfig?
    var modelContainer: ModelContainer?
    var isLoading = false
    var downloadProgress: Double = 0
    var loadingModelName: String = ""
    var errorMessage: String?

    private var idleTimer: Timer?
    private(set) var lastUsed: Date?

    /// Load a model, unloading the current one first.
    /// Prefers the local snapshot from ~/.cache/huggingface/hub/ (shared with the Python server).
    /// Only downloads if the model isn't cached locally.
    func loadModel(_ config: ModelConfig) async {
        if currentModel?.id == config.id && modelContainer != nil {
            return // already loaded
        }

        unloadModel()
        isLoading = true
        downloadProgress = 0
        loadingModelName = config.displayName
        errorMessage = nil

        do {
            let container: ModelContainer
            let progressHandler: @Sendable (Progress) -> Void = { progress in
                Task { @MainActor in
                    self.downloadProgress = progress.fractionCompleted
                }
            }

            let configuration: ModelConfiguration
            if let localDir = LocalModelResolver.resolve(repoId: config.repoId) {
                configuration = ModelConfiguration(directory: localDir)
            } else {
                configuration = config.modelConfiguration
            }

            container = try await VLMModelFactory.shared.loadContainer(
                configuration: configuration,
                progressHandler: progressHandler
            )

            self.modelContainer = container
            self.currentModel = config
            touchActivity()
        } catch {
            self.errorMessage = "Failed to load model: \(error.localizedDescription)"
        }

        isLoading = false
    }

    /// Unload the current model and free GPU memory.
    func unloadModel() {
        idleTimer?.invalidate()
        idleTimer = nil
        lastUsed = nil
        modelContainer = nil
        currentModel = nil
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
