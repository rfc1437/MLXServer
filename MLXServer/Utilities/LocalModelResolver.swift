import Foundation

/// Resolves HuggingFace model repos to local directories.
///
/// HubApi(downloadBase: .cachesDirectory, cache: nil) downloads models to:
///   ~/Library/Containers/de.rfc1437.mlxserver/Data/Library/Caches/models/{org}/{name}/
enum LocalModelResolver {

    /// Base directory where HubApi stores downloaded models.
    private static let modelsBase: URL? = {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("models", isDirectory: true)
    }()

    /// Resolve a HuggingFace repo ID (e.g. "mlx-community/gemma-3-4b-it-4bit")
    /// to its local directory, if it exists.
    ///
    /// Returns `nil` if the model hasn't been downloaded yet.
    static func resolve(repoId: String) -> URL? {
        guard let base = modelsBase else { return nil }
        let modelDir = base.appendingPathComponent(repoId, isDirectory: true)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir), isDir.boolValue {
            return modelDir
        }
        return nil
    }

    /// Check if a model is available locally.
    static func isAvailable(repoId: String) -> Bool {
        resolve(repoId: repoId) != nil
    }

    /// Delete the local cache for a model so it will be re-downloaded next time.
    @discardableResult
    static func deleteLocal(repoId: String) -> Bool {
        guard let base = modelsBase else { return false }
        let modelDir = base.appendingPathComponent(repoId, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return false }
        do {
            try FileManager.default.removeItem(at: modelDir)
            print("[LocalModelResolver] Deleted \(modelDir.path)")
            return true
        } catch {
            print("[LocalModelResolver] Failed to delete \(modelDir.path): \(error)")
            return false
        }
    }
}
