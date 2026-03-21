import Foundation

/// Resolves HuggingFace model repos to local directories.
///
/// HubApi(downloadBase: .cachesDirectory, cache: nil) downloads models to:
///   ~/Library/Containers/de.rfc1437.mlxserver/Data/Library/Caches/models/{org}/{name}/
enum LocalModelResolver {

    struct LocalModelInfo: Identifiable, Hashable {
        let repoId: String
        let directory: URL
        let sizeBytes: Int64
        let contextLength: Int
        let loaderKinds: [ModelConfig.LoaderKind]
        let supportsImages: Bool

        var id: String { repoId }
    }

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

    static func discoveredLocalModels() -> [LocalModelInfo] {
        guard let base = modelsBase else { return [] }
        return discoverModels(in: base)
    }

    static func discoverModels(in base: URL) -> [LocalModelInfo] {
        let fileManager = FileManager.default
        let directoryKeys: Set<URLResourceKey> = [.isDirectoryKey]
        guard let ownerDirectories = try? fileManager.contentsOfDirectory(
            at: base,
            includingPropertiesForKeys: Array(directoryKeys),
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var discovered: [LocalModelInfo] = []

        for ownerDirectory in ownerDirectories {
            guard isDirectory(ownerDirectory) else { continue }
            guard let repoDirectories = try? fileManager.contentsOfDirectory(
                at: ownerDirectory,
                includingPropertiesForKeys: Array(directoryKeys),
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for repoDirectory in repoDirectories where isDirectory(repoDirectory) {
                if let info = localModelInfo(ownerDirectory: ownerDirectory, repoDirectory: repoDirectory) {
                    discovered.append(info)
                }
            }
        }

        return discovered.sorted {
            $0.repoId.localizedCaseInsensitiveCompare($1.repoId) == .orderedAscending
        }
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

    private static func localModelInfo(ownerDirectory: URL, repoDirectory: URL) -> LocalModelInfo? {
        let repoId = "\(ownerDirectory.lastPathComponent)/\(repoDirectory.lastPathComponent)"
        guard containsModelArtifacts(at: repoDirectory) else { return nil }

        let config = readJSONObject(at: repoDirectory.appendingPathComponent("config.json"))
        let tokenizerConfig = readJSONObject(at: repoDirectory.appendingPathComponent("tokenizer_config.json"))
        let supportsImages = inferredSupportsImages(
            repoDirectory: repoDirectory,
            config: config,
            tokenizerConfig: tokenizerConfig
        )
        let sizeBytes = directorySize(at: repoDirectory)
        let contextLength = inferredContextLength(config: config, tokenizerConfig: tokenizerConfig)
        let loaderKinds: [ModelConfig.LoaderKind] = supportsImages ? [.vlm, .llm] : [.llm, .vlm]

        return LocalModelInfo(
            repoId: repoId,
            directory: repoDirectory,
            sizeBytes: sizeBytes,
            contextLength: contextLength,
            loaderKinds: loaderKinds,
            supportsImages: supportsImages
        )
    }

    private static func containsModelArtifacts(at directory: URL) -> Bool {
        let requiredPaths = [
            directory.appendingPathComponent("config.json").path,
            directory.appendingPathComponent("model.safetensors").path,
            directory.appendingPathComponent("model.safetensors.index.json").path,
        ]
        return requiredPaths.contains { FileManager.default.fileExists(atPath: $0) }
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private static func readJSONObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func inferredSupportsImages(
        repoDirectory: URL,
        config: [String: Any]?,
        tokenizerConfig: [String: Any]?
    ) -> Bool {
        if config?["vision_config"] != nil {
            return true
        }
        if tokenizerConfig?["image_token"] != nil {
            return true
        }

        let metadataFiles = [
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        ]
        return metadataFiles.contains {
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent($0).path)
        }
    }

    private static func inferredContextLength(
        config: [String: Any]?,
        tokenizerConfig: [String: Any]?
    ) -> Int {
        if let value = integerValue(at: ["text_config", "max_position_embeddings"], in: config) {
            return value
        }
        if let value = integerValue(at: ["max_position_embeddings"], in: config) {
            return value
        }
        if let value = integerValue(at: ["model_max_length"], in: tokenizerConfig) {
            return value
        }
        return 0
    }

    private static func integerValue(at path: [String], in json: [String: Any]?) -> Int? {
        guard let json else { return nil }

        var current: Any = json
        for component in path {
            guard let dictionary = current as? [String: Any], let next = dictionary[component] else {
                return nil
            }
            current = next
        }

        if let number = current as? NSNumber {
            return number.intValue
        }
        return current as? Int
    }

    private static func directorySize(at directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: Set(keys)), values.isRegularFile == true else {
                continue
            }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }
}
