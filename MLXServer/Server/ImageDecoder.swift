import AppKit
import CoreImage
import Foundation
import MLXLMCommon

/// Extracted from APIServer — decodes data URIs to UserInput.Image.
enum ImageDecoder {
    struct DecodedImage {
        let image: UserInput.Image
        let estimatedBytes: Int
    }

    static func decode(_ urlString: String) -> DecodedImage? {
        let base64String: String
        if urlString.hasPrefix("data:") {
            guard let commaIndex = urlString.firstIndex(of: ",") else { return nil }
            base64String = String(urlString[urlString.index(after: commaIndex)...])
        } else {
            base64String = urlString
        }

        guard let data = Data(base64Encoded: base64String),
              let nsImage = NSImage(data: data),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        let estimatedBytes = max(data.count, cgImage.width * cgImage.height * 4)
        return DecodedImage(image: .ciImage(CIImage(cgImage: cgImage)), estimatedBytes: estimatedBytes)
    }
}