import MLXLMCommon
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

    func testDecodeJPEGDataURI() {
        let image = ImageDecoder.decode(TestImageFixtures.primaryJPEGDataURI)

        XCTAssertNotNil(image)
        XCTAssertGreaterThanOrEqual(image?.estimatedBytes ?? 0, 64 * 64 * 4)
    }

    func testDecodeLarge4KDataURI() throws {
        let image = try XCTUnwrap(ImageDecoder.decode(TestImageFixtures.largeDataURI))

        XCTAssertGreaterThanOrEqual(image.estimatedBytes, 4_096 * 4_096 * 4)

        if case .ciImage(let ciImage) = image.image {
            XCTAssertEqual(Int(ciImage.extent.width), 4_096)
            XCTAssertEqual(Int(ciImage.extent.height), 4_096)
        } else {
            XCTFail("Expected CIImage-backed decoded image")
        }
    }
}