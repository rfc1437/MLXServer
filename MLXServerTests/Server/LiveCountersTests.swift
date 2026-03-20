import Foundation
import XCTest
@testable import MLX_Server

final class LiveCountersTests: XCTestCase {
    override func tearDown() {
        LiveCounters.shared.reset()
        super.tearDown()
    }

    func testTracksRequestMetricsAndDeduplicatesDisconnects() {
        let requestId = "req-1"

        LiveCounters.shared.reset()
        LiveCounters.shared.requestStarted(requestId: requestId, contextLength: 8_192)
        LiveCounters.shared.requestPhaseChanged(requestId: requestId, phase: .prefilling)
        LiveCounters.shared.recordPrefillReuse(requestId: requestId, matchedPromptTokens: 40, promptTokenCount: 64)
        LiveCounters.shared.visionProcessingCompleted(requestId: requestId, duration: 0.25)

        Thread.sleep(forTimeInterval: 0.01)
        LiveCounters.shared.prefillCompleted(requestId: requestId, promptTokens: 64)

        Thread.sleep(forTimeInterval: 0.01)
        LiveCounters.shared.firstTokenGenerated(requestId: requestId)
        LiveCounters.shared.tokenGenerated(tokensPerSecond: 12.5, totalGenerated: 3)
        LiveCounters.shared.disconnectDetected(requestId: requestId)
        LiveCounters.shared.disconnectDetected(requestId: requestId)

        let inFlight = LiveCounters.shared.snapshot()
        XCTAssertEqual(inFlight.cacheMatchDepth, 40)
        XCTAssertEqual(inFlight.currentCacheMatchedPromptTokens, 40)
        XCTAssertEqual(inFlight.currentCacheRebuiltPromptTokens, 24)
        XCTAssertEqual(inFlight.visionEncoderTime, 0.25, accuracy: 0.0001)
        XCTAssertGreaterThan(inFlight.prefillTokensPerSecond, 0)
        XCTAssertGreaterThan(inFlight.timeToFirstToken, 0)
        XCTAssertEqual(inFlight.totalDisconnects, 1)

        LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: 3)

        let completed = LiveCounters.shared.snapshot()
        XCTAssertEqual(completed.totalPromptTokens, 64)
        XCTAssertEqual(completed.totalGenerationTokens, 3)
        XCTAssertEqual(completed.totalVisionEncoderDuration, 0.25, accuracy: 0.0001)
        XCTAssertEqual(completed.totalDisconnects, 1)
    }
}