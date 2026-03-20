import XCTest
@testable import MLX_Server

final class ImageDecoderTests: XCTestCase {
    private let onePixelPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lRyXWQAAAABJRU5ErkJggg=="

    func testDecodeDataURI() {
        let image = ImageDecoder.decode("data:image/png;base64,\(onePixelPNGBase64)")

        XCTAssertNotNil(image)
        XCTAssertGreaterThanOrEqual(image?.estimatedBytes ?? 0, 4)
    }

    func testDecodePlainBase64() {
        let image = ImageDecoder.decode(onePixelPNGBase64)

        XCTAssertNotNil(image)
        XCTAssertGreaterThanOrEqual(image?.estimatedBytes ?? 0, 4)
    }
}