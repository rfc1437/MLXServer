import Foundation
import MLX
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

    func testCheckoutHitDoesNotCountAsEviction() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 3, 4], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(snapshot.totalEvictions, 0)
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

    func testSupersequenceLookupReusesLongerEntryForShorterQuery() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let entryId = UUID()
        cache.store(entryId: entryId, kvCache: [], cacheKey: [1, 2, 3, 4], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 3], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, entryId)
        XCTAssertEqual(lease.matchedTokenCount, 3)
        XCTAssertEqual(snapshot.totalHits, 1)
        XCTAssertEqual(snapshot.supersequenceHits, 1)
        XCTAssertEqual(snapshot.prefixHits, 0)
        XCTAssertEqual(snapshot.lcpHits, 0)
    }

    func testLCPLookupReusesSharedPrefixAcrossDivergentSuffixes() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let entryId = UUID()
        cache.store(entryId: entryId, kvCache: [], cacheKey: [10, 20, 90], modelId: "model")

        let lease = cache.lookup(cacheKey: [10, 20, 30], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, entryId)
        XCTAssertEqual(lease.matchedTokenCount, 2)
        XCTAssertEqual(snapshot.totalHits, 1)
        XCTAssertEqual(snapshot.lcpHits, 1)
        XCTAssertEqual(snapshot.prefixHits, 0)
        XCTAssertEqual(snapshot.supersequenceHits, 0)
    }

    func testLCPLookupRejectsShallowSharedPrefix() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        cache.store(entryId: UUID(), kvCache: [], cacheKey: [10, 20, 30, 40], modelId: "model")

        let lease = cache.lookup(cacheKey: [10, 99, 98, 97], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertFalse(lease.isHit)
        XCTAssertEqual(lease.matchedTokenCount, 0)
        XCTAssertEqual(snapshot.totalHits, 0)
        XCTAssertEqual(snapshot.totalMisses, 1)
        XCTAssertEqual(snapshot.lcpHits, 0)
    }

    func testLookupPrefersPrefixMatchOverSupersequenceAndLCP() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let prefixId = UUID()
        cache.store(entryId: prefixId, kvCache: [], cacheKey: [7, 8], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [7, 8, 9, 10], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [7, 8, 11], modelId: "model")

        let lease = cache.lookup(cacheKey: [7, 8, 12], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, prefixId)
        XCTAssertEqual(lease.matchedTokenCount, 2)
        XCTAssertEqual(snapshot.prefixHits, 1)
        XCTAssertEqual(snapshot.supersequenceHits, 0)
        XCTAssertEqual(snapshot.lcpHits, 0)
    }

    func testSupersequenceSkipsNonTrimmableLayersGracefully() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let layer = TestTrimRecordingCache(offset: 4, trimmable: false)
        cache.store(entryId: UUID(), kvCache: [layer], cacheKey: [1, 2, 3, 4], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 3], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertFalse(lease.isHit)
        XCTAssertEqual(layer.offset, 4)
        XCTAssertTrue(layer.trimCalls.isEmpty)
        XCTAssertEqual(snapshot.supersequenceHits, 0)
        XCTAssertEqual(snapshot.totalMisses, 1)
    }

    func testSupersequenceChoosesShallowestCandidate() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let shallowestId = UUID()
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3, 4, 5], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3, 4], modelId: "model")
        cache.store(entryId: shallowestId, kvCache: [], cacheKey: [1, 2, 3], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2], modelId: "model")

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, shallowestId)
        XCTAssertEqual(lease.matchedTokenCount, 2)
    }

    func testSupersequencePathWinsWhenFullQueryWalkCanAlsoSeeDivergentSibling() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let supersequenceId = UUID()
        cache.store(entryId: supersequenceId, kvCache: [], cacheKey: [1, 2, 3], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 9, 8], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2], modelId: "model")
        let snapshot = cache.snapshot()

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, supersequenceId)
        XCTAssertEqual(snapshot.supersequenceHits, 1)
        XCTAssertEqual(snapshot.lcpHits, 0)
    }

    func testLCPChoosesShallowestSiblingCandidate() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let shallowestId = UUID()
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 3, 7], modelId: "model")
        cache.store(entryId: UUID(), kvCache: [], cacheKey: [1, 2, 4, 7, 8], modelId: "model")
        cache.store(entryId: shallowestId, kvCache: [], cacheKey: [1, 2, 5], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 9, 9], modelId: "model")

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.entryId, shallowestId)
        XCTAssertEqual(lease.matchedTokenCount, 2)
    }

    func testTrimUsesExactExcessAndReducesOffset() {
        let cache = TokenPrefixCache(
            memoryBudgetBytes: 10_000,
            estimateBytesProvider: { _ in 1_024 }
        )

        let layer = TestTrimRecordingCache(offset: 5, trimmable: true)
        cache.store(entryId: UUID(), kvCache: [layer], cacheKey: [1, 2, 3, 4, 5], modelId: "model")

        let lease = cache.lookup(cacheKey: [1, 2, 3], modelId: "model")

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(layer.trimCalls, [2])
        XCTAssertEqual(layer.offset, 3)
    }

    func testComputeMemoryBudgetUsesFallbackWhenDeviceUnavailable() {
        let budget = TokenPrefixCache.computeMemoryBudget(recommendedWorkingSetSize: nil)

        XCTAssertEqual(budget, 512 * 1024 * 1024)
    }

    func testComputeMemoryBudgetClampsToMinimumFloor() {
        let budget = TokenPrefixCache.computeMemoryBudget(recommendedWorkingSetSize: 512 * 1024 * 1024)

        XCTAssertEqual(budget, 256 * 1024 * 1024)
    }

    func testComputeMemoryBudgetUsesTwentyPercentOfWorkingSet() {
        let budget = TokenPrefixCache.computeMemoryBudget(recommendedWorkingSetSize: 8 * 1024 * 1024 * 1024)

        XCTAssertEqual(budget, Int(Double(8 * 1024 * 1024 * 1024) * 0.20))
    }

    func testComputeMemoryBudgetClampsToMaximumCap() {
        let budget = TokenPrefixCache.computeMemoryBudget(recommendedWorkingSetSize: 80 * 1024 * 1024 * 1024)

        XCTAssertEqual(budget, 8 * 1024 * 1024 * 1024)
    }
}

private final class TestTrimRecordingCache: KVCache {
    private var arrays: [MLXArray] = []
    var offset: Int
    let maxSize: Int? = nil
    let trimmable: Bool
    private(set) var trimCalls: [Int] = []

    init(offset: Int, trimmable: Bool) {
        self.offset = offset
        self.trimmable = trimmable
    }

    func innerState() -> [MLXArray] {
        arrays
    }

    var state: [MLXArray] {
        get { arrays }
        set { arrays = newValue }
    }

    var metaState: [String] {
        get { [String(offset)] }
        set { offset = Int(newValue.first ?? "0") ?? 0 }
    }

    var isTrimmable: Bool { trimmable }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("TestTrimRecordingCache does not support update")
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        guard trimmable else { return 0 }
        trimCalls.append(n)
        offset = max(0, offset - n)
        return n
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
}