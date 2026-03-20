import XCTest
@testable import MLX_Server

final class CancellationTokenTests: XCTestCase {
    func testStartsNotCancelled() {
        let token = CancellationToken()

        XCTAssertFalse(token.isCancelled)
    }

    func testCancelSetsFlag() {
        let token = CancellationToken()

        token.cancel()

        XCTAssertTrue(token.isCancelled)
    }
}