import XCTest
@testable import MLX_Server

final class ImageDecoderTests: XCTestCase {
    func testDecodeDataURI() {
        let image = ImageDecoder.decode(TestImageFixtures.primaryDataURI)

        XCTAssertNotNil(image)
        XCTAssertGreaterThanOrEqual(image?.estimatedBytes ?? 0, 4)
    }

    func testDecodePlainBase64() {
        let image = ImageDecoder.decode(TestImageFixtures.primaryPNGBase64)

        XCTAssertNotNil(image)
        XCTAssertGreaterThanOrEqual(image?.estimatedBytes ?? 0, 4)
    }
}