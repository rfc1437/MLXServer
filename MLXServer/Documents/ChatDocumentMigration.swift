import Foundation

enum ChatDocumentMigration {
    private struct ManifestEnvelope: Decodable {
        let schemaVersion: Int
    }

    static func loadManifest(from data: Data) throws -> ChatDocumentManifest {
        let decoder = JSONDecoder.chatDocumentDecoder
        let envelope = try decoder.decode(ManifestEnvelope.self, from: data)

        switch envelope.schemaVersion {
        case 1:
            return try decoder.decode(ChatDocumentManifest.self, from: data)
        default:
            throw ChatDocumentError.unsupportedSchemaVersion(envelope.schemaVersion)
        }
    }
}
