import Foundation

/// Resolves HuggingFace model repos to local snapshot directories,
/// matching the cache layout used by Python's `huggingface_hub`.
///
/// Checks two locations:
///   1. App sandbox container: ~/Library/Containers/com.mlxserver.app/.../huggingface/hub/
///   2. System-wide cache: ~/.cache/huggingface/hub/ (shared with Python tools)
///
/// Cache structure:
///   .../huggingface/hub/models--{org}--{name}/snapshots/{hash}/
enum LocalModelResolver {

    /// All HuggingFace cache directories to search, in priority order.
    /// The sandboxed container path is checked first (where the app downloads to),
    /// then the system-wide Python cache (for models downloaded via huggingface-cli).
    private static let cacheBases: [URL] = {
        var bases: [URL] = []

        // 1. Sandboxed app container cache (where swift-transformers Hub downloads to)
        let containerCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/huggingface/hub", isDirectory: true)
        bases.append(containerCache)

        // 2. System-wide ~/.cache/huggingface/hub/ (Python huggingface_hub)
        //    When sandboxed, homeDirectory points to the container, so construct the real path.
        let realHome = URL(fileURLWithPath: NSHomeDirectory())
        let systemCache = realHome
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        // Avoid duplicate if they resolve to the same path
        if systemCache.path != containerCache.path {
            bases.append(systemCache)
        }

        // 3. Also try the unsandboxed home directory path
        let globalHome = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        if globalHome.path != containerCache.path && globalHome.path != systemCache.path {
            bases.append(globalHome)
        }

        return bases
    }()

    /// Resolve a HuggingFace repo ID (e.g. "mlx-community/gemma-3-4b-it-4bit")
    /// to its local snapshot directory, if it exists.
    ///
    /// Returns `nil` if the model hasn't been downloaded yet.
    static func resolve(repoId: String) -> URL? {
        let dirName = "models--" + repoId.replacingOccurrences(of: "/", with: "--")

        for cacheBase in cacheBases {
            let snapshotsDir = cacheBase
                .appendingPathComponent(dirName, isDirectory: true)
                .appendingPathComponent("snapshots", isDirectory: true)

            guard let contents = try? FileManager.default.contentsOfDirectory(
                at: snapshotsDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            if let snapshot = contents
                .filter({ (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true })
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                .last {
                return snapshot
            }
        }

        return nil
    }

    /// Check if a model is available locally.
    static func isAvailable(repoId: String) -> Bool {
        resolve(repoId: repoId) != nil
    }

    /// Delete the local cache for a model so it will be re-downloaded next time.
    /// Removes from all cache locations.
    /// Returns true if something was deleted.
    @discardableResult
    static func deleteLocal(repoId: String) -> Bool {
        let dirName = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        var deleted = false

        for cacheBase in cacheBases {
            let modelDir = cacheBase.appendingPathComponent(dirName, isDirectory: true)
            guard FileManager.default.fileExists(atPath: modelDir.path) else { continue }
            do {
                try FileManager.default.removeItem(at: modelDir)
                print("[LocalModelResolver] Deleted \(modelDir.path)")
                deleted = true
            } catch {
                print("[LocalModelResolver] Failed to delete \(modelDir.path): \(error)")
            }
        }

        // Also clean up the per-model cache in the container (used by swift-transformers)
        let containerModelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/models", isDirectory: true)
            .appendingPathComponent(repoId, isDirectory: true)
        if FileManager.default.fileExists(atPath: containerModelsDir.path) {
            do {
                try FileManager.default.removeItem(at: containerModelsDir)
                print("[LocalModelResolver] Deleted \(containerModelsDir.path)")
                deleted = true
            } catch {
                print("[LocalModelResolver] Failed to delete \(containerModelsDir.path): \(error)")
            }
        }

        return deleted
    }
}
