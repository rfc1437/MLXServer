import Foundation

/// Resolves HuggingFace model repos to local snapshot directories,
/// matching the cache layout used by Python's `huggingface_hub`.
///
/// Cache structure:
///   ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/
enum LocalModelResolver {

    /// The standard HuggingFace cache directory used by Python's `huggingface_hub`.
    private static let cacheBase: URL = {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }()

    /// Resolve a HuggingFace repo ID (e.g. "mlx-community/gemma-3-4b-it-4bit")
    /// to its local snapshot directory, if it exists.
    ///
    /// Returns `nil` if the model hasn't been downloaded yet.
    static func resolve(repoId: String) -> URL? {
        // Convert "mlx-community/gemma-3-4b-it-4bit" → "models--mlx-community--gemma-3-4b-it-4bit"
        let dirName = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        let snapshotsDir = cacheBase
            .appendingPathComponent(dirName, isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)

        // Find the first (usually only) snapshot hash directory
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        // Return the most recent snapshot (last alphabetically = latest hash)
        return contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .last
    }

    /// Check if a model is available locally.
    static func isAvailable(repoId: String) -> Bool {
        resolve(repoId: repoId) != nil
    }
}
