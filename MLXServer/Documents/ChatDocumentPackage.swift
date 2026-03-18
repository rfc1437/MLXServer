import Foundation
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let mlxChatDocument = UTType(exportedAs: "de.rfc1437.mlxserver.chat", conformingTo: .package)
}

enum ChatDocumentError: LocalizedError {
    case invalidPackage
    case missingManifest
    case missingAttachment(String)
    case invalidAttachmentData(String)
    case unsupportedSchemaVersion(Int)
    case saveWhileGenerating

    var errorDescription: String? {
        switch self {
        case .invalidPackage:
            return "The selected file is not a valid MLX Server chat document."
        case .missingManifest:
            return "The chat document is missing manifest.json."
        case .missingAttachment(let path):
            return "The chat document is missing attachment \(path)."
        case .invalidAttachmentData(let path):
            return "The attachment \(path) could not be decoded as an image."
        case .unsupportedSchemaVersion(let version):
            return "This chat document uses unsupported schema version \(version)."
        case .saveWhileGenerating:
            return "Stop generation before saving this chat document."
        }
    }
}

struct ChatDocumentPackage: FileDocument {
    static var readableContentTypes: [UTType] { [.mlxChatDocument] }
    static var writableContentTypes: [UTType] { [.mlxChatDocument] }

    let manifest: ChatDocumentManifest
    let attachmentContents: [String: Data]

    init(manifest: ChatDocumentManifest, attachmentContents: [String: Data]) {
        self.manifest = manifest
        self.attachmentContents = attachmentContents
    }

    init(contentsOf url: URL) throws {
        let wrapper = try FileWrapper(url: url, options: .immediate)
        try self.init(rootWrapper: wrapper)
    }

    init(configuration: ReadConfiguration) throws {
        try self.init(rootWrapper: configuration.file)
    }

    private init(rootWrapper: FileWrapper) throws {
        guard let fileWrappers = rootWrapper.fileWrappers else {
            throw ChatDocumentError.invalidPackage
        }

        guard let manifestWrapper = fileWrappers["manifest.json"],
              let manifestData = manifestWrapper.regularFileContents else {
            throw ChatDocumentError.missingManifest
        }

        let manifest = try ChatDocumentMigration.loadManifest(from: manifestData)
        var attachmentContents: [String: Data] = [:]

        for message in manifest.messages {
            for attachment in message.attachments {
                if attachmentContents[attachment.relativePath] != nil {
                    continue
                }

                let pathComponents = attachment.relativePath.split(separator: "/").map(String.init)
                guard let attachmentData = Self.data(at: pathComponents, from: fileWrappers) else {
                    throw ChatDocumentError.missingAttachment(attachment.relativePath)
                }
                attachmentContents[attachment.relativePath] = attachmentData
            }
        }

        self.manifest = manifest
        self.attachmentContents = attachmentContents
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        try makeFileWrapper()
    }

    func write(to url: URL) throws {
        let wrapper = try makeFileWrapper()
        let fileManager = FileManager.default
        let options = FileWrapper.WritingOptions.atomic
        if fileManager.fileExists(atPath: url.path) {
            try wrapper.write(to: url, options: options, originalContentsURL: url)
        } else {
            try wrapper.write(to: url, options: options, originalContentsURL: nil)
        }
    }

    private func makeFileWrapper() throws -> FileWrapper {
        var wrappers: [String: FileWrapper] = [:]
        let manifestData = try JSONEncoder.chatDocumentEncoder.encode(manifest)
        wrappers["manifest.json"] = FileWrapper(regularFileWithContents: manifestData)

        var attachmentWrappers: [String: FileWrapper] = [:]
        for (relativePath, data) in attachmentContents {
            let pathComponents = relativePath.split(separator: "/").map(String.init)
            guard pathComponents.count == 2, pathComponents.first == "attachments" else { continue }
            attachmentWrappers[pathComponents[1]] = FileWrapper(regularFileWithContents: data)
        }

        wrappers["attachments"] = FileWrapper(directoryWithFileWrappers: attachmentWrappers)
        return FileWrapper(directoryWithFileWrappers: wrappers)
    }

    private static func data(at pathComponents: [String], from wrappers: [String: FileWrapper]) -> Data? {
        guard let first = pathComponents.first else { return nil }
        guard let wrapper = wrappers[first] else { return nil }

        if pathComponents.count == 1 {
            return wrapper.regularFileContents
        }

        guard let childWrappers = wrapper.fileWrappers else { return nil }
        return data(at: Array(pathComponents.dropFirst()), from: childWrappers)
    }
}

private extension JSONEncoder {
    static var chatDocumentEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

extension JSONDecoder {
    static var chatDocumentDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
