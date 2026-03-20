import Foundation

enum TestImageFixtures {
    private static let repoRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }()

    private static func loadBase64(named name: String) -> String {
        let url = repoRoot
            .appendingPathComponent("MLXServer")
            .appendingPathComponent("Assets.xcassets")
            .appendingPathComponent("AppIcon.appiconset")
            .appendingPathComponent(name)

        guard let data = try? Data(contentsOf: url) else {
            fatalError("Missing image fixture at \(url.path)")
        }

        return data.base64EncodedString()
    }

    static let primaryPNGBase64 = loadBase64(named: "icon_16x16.png")
    static let alternatePNGBase64 = loadBase64(named: "icon_32x32.png")

    static let primaryDataURI = "data:image/png;base64,\(primaryPNGBase64)"
    static let alternateDataURI = "data:image/png;base64,\(alternatePNGBase64)"
}