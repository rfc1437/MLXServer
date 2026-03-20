import Foundation
import XCTest
import MLXLMCommon
@testable import MLX_Server

final class TokenPrefixCacheTests: XCTestCase {
    func testStoreAndLookupRemovesCheckedOutEntry() {
        var now = Date(timeIntervalSince1970: 100)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 },
            nowProvider: { now }
        )

        let entryId = UUID()
        cache.store(entryId: entryId, kvCache: [], cacheKey: [1, 2, 3], modelId: "model")

        XCTAssertEqual(cache.snapshot().totalEntries, 1)

        let lease = cache.lookup(cacheKey: [1, 2, 3, 4], modelId: "model")

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, entryId)
        XCTAssertEqual(lease.matchedTokenCount, 3)
        XCTAssertNotNil(lease.kvCache)
        XCTAssertEqual(cache.snapshot().totalEntries, 0)
    }

    func testLookupPrefersDeepestPrefixMatch() {
        var now = Date(timeIntervalSince1970: 100)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 },
            nowProvider: { now }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2], modelId: "model")
        now.addTimeInterval(1)
        let deepId = UUID()
        cache.store(entryId: deepId, kvCache: [], cacheKey: [1, 2, 3], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 3, 4], modelId: "model")

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, deepId)
        XCTAssertEqual(lease.matchedTokenCount, 3)
    }

    func testEvictsLeastRecentlyUsedEntryWhenOverBudget() {
        var now = Date(timeIntervalSince1970: 100)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 2_048,
            estimateBytesProvider: { _ in 1_024 },
            nowProvider: { now }
        )

        let firstId = UUID()
        cache.store(entryId: firstId, kvCache: [], cacheKey: [1], modelId: "model")
        now.addTimeInterval(1)
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [2], modelId: "model")
        now.addTimeInterval(1)
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [3], modelId: "model")

        let firstLookup = cache.lookup(cacheKey: [1], modelId: "model")
        let secondLookup = cache.lookup(cacheKey: [2], modelId: "model")
        let thirdLookup = cache.lookup(cacheKey: [3], modelId: "model")

        XCTAssertFalse(firstLookup.isHit)
        XCTAssertTrue(secondLookup.isHit)
        XCTAssertTrue(thirdLookup.isHit)
    }

    func testSnapshotPrunesExpiredEntries() {
        var now = Date(timeIntervalSince1970: 100)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            idleTTL: 5,
            estimateBytesProvider: { _ in 1_024 },
            nowProvider: { now }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3], modelId: "model")
        XCTAssertEqual(cache.snapshot().totalEntries, 1)

        now.addTimeInterval(10)
        let snapshot = cache.snapshot()

        XCTAssertEqual(snapshot.totalEntries, 0)
        XCTAssertGreaterThanOrEqual(snapshot.totalEvictions, 1)
    }

    func testLookupPrunesTrieNodesForRemovedBranch() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 4], modelId: "model")

        XCTAssertEqual(cache.debugTrieNodeCount(), 5)

        _ = cache.lookup(cacheKey: [1, 2, 3], modelId: "model")

        XCTAssertEqual(cache.debugTrieNodeCount(), 4)

        _ = cache.lookup(cacheKey: [1, 2, 4], modelId: "model")

        XCTAssertEqual(cache.debugTrieNodeCount(), 1)
    }

    func testSnapshotReportsHitRateAndTokenTotals() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 2_048 }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [10, 20, 30], modelId: "model")
        _ = cache.lookup(cacheKey: [10, 20, 30, 40], modelId: "model")
        _ = cache.lookup(cacheKey: [99], modelId: "model")

        let snapshot = cache.snapshot()

        XCTAssertEqual(snapshot.totalHits, 1)
        XCTAssertEqual(snapshot.totalMisses, 1)
        XCTAssertEqual(snapshot.hitRate, 50, accuracy: 0.001)
        XCTAssertEqual(snapshot.totalCachedTokens, 0)
        XCTAssertEqual(snapshot.estimatedBytes, 0)
    }
}