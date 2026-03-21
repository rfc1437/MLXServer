import AppKit
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

    private static func generatedBitmapData(
        width: Int,
        height: Int,
        fileType: NSBitmapImageRep.FileType,
        compressionFactor: Double? = nil
    ) -> Data {
        let bytesPerRow = width * 4
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: bytesPerRow,
            bitsPerPixel: 32
        ) else {
            fatalError("Failed to create bitmap fixture")
        }

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        let imageRect = NSRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
        NSColor(calibratedRed: 0.18, green: 0.45, blue: 0.87, alpha: 1).setFill()
        imageRect.fill()
        NSColor.white.setStroke()
        let inset = CGFloat(max(8, min(width, height) / 16))
        NSBezierPath(rect: imageRect.insetBy(dx: inset, dy: inset)).stroke()
        NSGraphicsContext.restoreGraphicsState()

        var properties: [NSBitmapImageRep.PropertyKey: Any] = [:]
        if let compressionFactor {
            properties[.compressionFactor] = compressionFactor
        }

        guard let data = rep.representation(using: fileType, properties: properties) else {
            fatalError("Failed to encode bitmap fixture")
        }

        return data
    }

    static let primaryPNGBase64 = loadBase64(named: "icon_16x16.png")
    static let alternatePNGBase64 = loadBase64(named: "icon_32x32.png")
    static let primaryJPEGBase64 = generatedBitmapData(
        width: 64,
        height: 64,
        fileType: .jpeg,
        compressionFactor: 0.85
    ).base64EncodedString()
    static let largePNGBase64 = generatedBitmapData(
        width: 4_096,
        height: 4_096,
        fileType: .png
    ).base64EncodedString()

    static let primaryDataURI = "data:image/png;base64,\(primaryPNGBase64)"
    static let alternateDataURI = "data:image/png;base64,\(alternatePNGBase64)"
    static let primaryJPEGDataURI = "data:image/jpeg;base64,\(primaryJPEGBase64)"
    static let largeDataURI = "data:image/png;base64,\(largePNGBase64)"
}