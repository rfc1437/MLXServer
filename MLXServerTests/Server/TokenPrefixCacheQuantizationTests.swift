import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLX_Server

final class TokenPrefixCacheQuantizationTests: XCTestCase {
    func testQuantizationConfigDefault() {
        let config = TokenPrefixCache.QuantizationConfig.default
        XCTAssertFalse(config.enabled)
        XCTAssertEqual(config.bits, 8)
        XCTAssertEqual(config.groupSize, 64)
        XCTAssertEqual(config.minTokens, 256)
    }

    func testQuantizationReducesStoredMemoryAndTracksSavings() {
        let rawCache = [makeSimpleCache(tokenCount: 320, heads: 4, headDim: 64)]
        let rawBytes = estimateBytes(rawCache)

        let cache = TokenPrefixCache(
            memoryBudgetBytes: rawBytes * 2,
            quantizationConfig: .aggressive
        )

        cache.store(
            entryId: UUID(),
            kvCache: rawCache,
            cacheKey: Array(1...320),
            modelId: "model"
        )

        let snapshot = cache.snapshot()

        XCTAssertTrue(snapshot.quantizationEnabled)
        XCTAssertGreaterThan(snapshot.quantizationBytesSaved, 0)
        XCTAssertLessThan(snapshot.estimatedBytes, rawBytes)
        XCTAssertLessThan(Double(snapshot.estimatedBytes) / Double(rawBytes), 0.80)
    }

    func testShortSequencesBelowThresholdRemainUnquantized() throws {
        let rawCache = [makeSimpleCache(tokenCount: 32)]
        let rawBytes = estimateBytes(rawCache)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: rawBytes * 2,
            quantizationConfig: .aggressive
        )

        cache.store(
            entryId: UUID(),
            kvCache: rawCache,
            cacheKey: Array(1...32),
            modelId: "model"
        )

        let snapshot = cache.snapshot()
        XCTAssertEqual(snapshot.quantizationBytesSaved, 0)
        XCTAssertEqual(snapshot.estimatedBytes, rawBytes)

        let lease = cache.lookup(cacheKey: Array(1...32), modelId: "model")
        let returned = try XCTUnwrap(lease.kvCache)
        XCTAssertTrue(returned.allSatisfy { $0 is KVCacheSimple })
        XCTAssertFalse(returned.contains { $0 is QuantizedKVCache })
    }

    func testQuantizedExactHitReturnsDequantizedCacheCloseToOriginal() throws {
        let rawCache = [makeSimpleCache(tokenCount: 300)]
        let cache = TokenPrefixCache(
            memoryBudgetBytes: estimateBytes(rawCache) * 2,
            quantizationConfig: .aggressive
        )

        cache.store(
            entryId: UUID(),
            kvCache: rawCache,
            cacheKey: Array(1...300),
            modelId: "model"
        )

        let lease = cache.lookup(cacheKey: Array(1...300), modelId: "model")
        let returned = try XCTUnwrap(lease.kvCache)

        XCTAssertTrue(lease.isHit)
        XCTAssertTrue(returned.allSatisfy { $0 is KVCacheSimple })
        XCTAssertFalse(returned.contains { $0 is QuantizedKVCache })
        XCTAssertEqual(returned.count, rawCache.count)

        for (original, roundTripped) in zip(rawCache, returned) {
            XCTAssertEqual(original.offset, roundTripped.offset)
            XCTAssertLessThanOrEqual(maxRelativeError(original.state[0], roundTripped.state[0]), 0.02)
            XCTAssertLessThanOrEqual(maxRelativeError(original.state[1], roundTripped.state[1]), 0.02)
        }
    }

    func testNonStandardLayersPassThroughUnquantized() throws {
        let nonStandard = NonStandardCache(tokenCount: 300, headDim: 32)
        let cache = TokenPrefixCache(
            memoryBudgetBytes: estimateBytes([nonStandard]) * 2,
            quantizationConfig: .aggressive
        )

        cache.store(
            entryId: UUID(),
            kvCache: [nonStandard],
            cacheKey: Array(1...300),
            modelId: "model"
        )

        let snapshot = cache.snapshot()
        XCTAssertEqual(snapshot.quantizationBytesSaved, 0)

        let lease = cache.lookup(cacheKey: Array(1...300), modelId: "model")
        let returned = try XCTUnwrap(lease.kvCache)
        XCTAssertEqual(returned.count, 1)
        XCTAssertTrue(returned[0] is NonStandardCache)
    }

    func testQuantizedSupersequenceHitReturnsDequantizedTrimmedCache() throws {
        let rawCache = [makeSimpleCache(tokenCount: 300)]
        let cache = TokenPrefixCache(
            memoryBudgetBytes: estimateBytes(rawCache) * 2,
            quantizationConfig: .aggressive
        )

        cache.store(
            entryId: UUID(),
            kvCache: rawCache,
            cacheKey: Array(1...300),
            modelId: "model"
        )

        let lease = cache.lookup(cacheKey: Array(1...260), modelId: "model")
        let returned = try XCTUnwrap(lease.kvCache)

        XCTAssertTrue(lease.isHit)
        XCTAssertEqual(lease.matchedTokenCount, 260)
        XCTAssertTrue(returned.allSatisfy { $0 is KVCacheSimple })
        for layer in returned {
            XCTAssertEqual(layer.offset, 260)
        }
    }

    func testQuantizationConfigChangesOnlyAffectFutureStores() {
        let firstCache = [makeSimpleCache(tokenCount: 300)]
        let secondCache = [makeSimpleCache(tokenCount: 300, base: 10_000)]
        let cache = TokenPrefixCache(
            memoryBudgetBytes: estimateBytes(firstCache) * 4,
            quantizationConfig: .default
        )

        cache.store(
            entryId: UUID(),
            kvCache: firstCache,
            cacheKey: Array(1...300),
            modelId: "model"
        )
        let before = cache.snapshot()
        XCTAssertEqual(before.quantizationBytesSaved, 0)

        cache.setQuantizationConfig(.aggressive)
        let toggled = cache.snapshot()
        XCTAssertTrue(toggled.quantizationEnabled)
        XCTAssertEqual(toggled.quantizationBytesSaved, 0)

        cache.store(
            entryId: UUID(),
            kvCache: secondCache,
            cacheKey: Array(1001...1300),
            modelId: "model"
        )

        let after = cache.snapshot()
        XCTAssertGreaterThan(after.quantizationBytesSaved, 0)
        XCTAssertGreaterThan(after.totalEntries, 1)
    }

    private func makeSimpleCache(tokenCount: Int, heads: Int = 2, headDim: Int = 64, base: Int = 0)
        -> KVCacheSimple
    {
        let count = heads * tokenCount * headDim
        let keyValues = (0..<count).map { index in
            Float(base + index) / Float(max(count - 1, 1)) * 2 - 1
        }
        let valueValues = keyValues.reversed()
        let keys = MLXArray(keyValues, [1, heads, tokenCount, headDim])
        let values = MLXArray(Array(valueValues), [1, heads, tokenCount, headDim])
        let cache = KVCacheSimple()
        cache.state = [keys, values]
        MLX.eval(cache.state)
        return cache
    }

    private func estimateBytes(_ cache: [KVCache]) -> Int {
        max(cache.flatMap(\.state).reduce(0) { $0 + $1.nbytes }, 1024)
    }

    private func maxRelativeError(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let left = lhs.asArray(Float.self)
        let right = rhs.asArray(Float.self)
        XCTAssertEqual(left.count, right.count)

        var maximum: Float = 0
        for (l, r) in zip(left, right) {
            let denominator = max(abs(l), 1e-6)
            maximum = max(maximum, abs(l - r) / denominator)
        }
        return maximum
    }
}

private final class NonStandardCache: KVCache {
    private var arrays: [MLXArray]
    var offset: Int
    let maxSize: Int? = nil

    init(tokenCount: Int, headDim: Int) {
        let count = tokenCount * headDim
        let values = (0..<count).map { Float($0) / Float(max(count - 1, 1)) }
        self.arrays = [MLXArray(values, [1, 1, tokenCount, headDim])]
        self.offset = tokenCount
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

    var isTrimmable: Bool { false }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("NonStandardCache is test-only and does not support update")
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
}
