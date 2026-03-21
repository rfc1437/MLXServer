import Foundation
import Metal
import MLX
import MLXLMCommon
import os

final class TokenPrefixCache: @unchecked Sendable {
    static let shared = TokenPrefixCache()

    struct CacheLease: @unchecked Sendable {
        let entryId: UUID
        let kvCache: [KVCache]?
        let matchedTokenCount: Int
        let isHit: Bool
    }

    struct EntrySummary: Identifiable, Sendable {
        let id: UUID
        let modelId: String
        let tokenCount: Int
        let estimatedBytes: Int
        let createdAt: Date
        let lastAccessAt: Date
        let hitCount: Int
    }

    struct Snapshot: Sendable {
        let totalEntries: Int
        let totalCachedTokens: Int
        let estimatedBytes: Int
        let memoryBudgetBytes: Int
        let memoryUsagePercent: Double
        let totalHits: Int
        let totalMisses: Int
        let totalEvictions: Int
        let hitRate: Double
        let prefixHits: Int
        let supersequenceHits: Int
        let lcpHits: Int
        let quantizationBytesSaved: Int  // Total bytes saved by quantization
        let quantizationEnabled: Bool
        let entries: [EntrySummary]
    }

    private final class TrieNode {
        var children: [Int: TrieNode] = [:]
        var entryId: UUID?
    }

    private struct CacheEntry {
        let id: UUID
        let modelId: String
        let kvCache: [KVCache]
        let tokenCount: Int
        let cacheKey: [Int]
        let estimatedBytes: Int
        let createdAt: Date
        var lastAccessAt: Date
        var hitCount: Int
        let isQuantized: Bool
    }

    private struct Stats {
        var totalHits: Int = 0
        var totalMisses: Int = 0
        var totalEvictions: Int = 0
        var totalPrefixHits: Int = 0
        var totalSupersequenceHits: Int = 0
        var totalLCPHits: Int = 0
        var totalQuantizationBytesSaved: Int = 0
    }

    struct QuantizationConfig: Sendable {
        /// Whether to quantize KV caches for storage
        let enabled: Bool
        /// Bit width for quantization (8 is recommended for 50% savings with minimal quality loss)
        let bits: Int
        /// Group size for quantization. Matches mlx-swift-lm default.
        let groupSize: Int
        /// Minimum token count before quantization applies. Short sequences don't benefit.
        let minTokens: Int

        static let `default` = QuantizationConfig(
            enabled: false,
            bits: 8,
            groupSize: 64,
            minTokens: 256
        )

        static let aggressive = QuantizationConfig(
            enabled: true,
            bits: 8,
            groupSize: 64,
            minTokens: 256
        )
    }

    private let lock = OSAllocatedUnfairLock()
    private let maxMemoryBytes: Int
    private let idleTTL: TimeInterval
    private let estimateBytesProvider: ([KVCache]) -> Int
    private let nowProvider: () -> Date
    private var root = TrieNode()
    private var entries: [UUID: CacheEntry] = [:]
    private var currentMemoryBytes: Int = 0
    private var stats = Stats()
    private var quantizationConfig: QuantizationConfig

    private init() {
        self.maxMemoryBytes = Self.computeMemoryBudget()
        self.idleTTL = 30 * 60
        self.estimateBytesProvider = Self.estimateBytes
        self.nowProvider = Date.init
        self.quantizationConfig = Self.preferencesQuantizationConfig()
    }

    init(
        memoryBudgetBytes: Int,
        idleTTL: TimeInterval = 30 * 60,
        estimateBytesProvider: @escaping ([KVCache]) -> Int = TokenPrefixCache.estimateBytes,
        nowProvider: @escaping () -> Date = Date.init,
        quantizationConfig: QuantizationConfig = .default
    ) {
        self.maxMemoryBytes = memoryBudgetBytes
        self.idleTTL = idleTTL
        self.estimateBytesProvider = estimateBytesProvider
        self.nowProvider = nowProvider
        self.quantizationConfig = quantizationConfig
    }

    /// Update quantization configuration.
    func setQuantizationConfig(_ config: QuantizationConfig) {
        lock.lock()
        self.quantizationConfig = config
        lock.unlock()
    }

    /// Get current quantization configuration.
    func getQuantizationConfig() -> QuantizationConfig {
        lock.lock()
        defer { lock.unlock() }
        return quantizationConfig
    }

    private static func preferencesQuantizationConfig() -> QuantizationConfig {
        guard Preferences.kvQuantizationEnabled else {
            return .default
        }

        return QuantizationConfig(
            enabled: true,
            bits: Preferences.kvQuantizationBits,
            groupSize: 64,
            minTokens: 256
        )
    }

    func lookup(cacheKey: [Int], modelId: String) -> CacheLease {
        lock.lock()
        let now = nowProvider()
        pruneExpiredLocked(now: now)
        let queryRealTokenCount = cacheKey.reduce(into: 0) { partialResult, token in
            if token >= 0 {
                partialResult += 1
            }
        }

        var node = root
        var bestMatch: (entryId: UUID, realTokenCount: Int)?
        var realTokenCount = 0
        var walkedFullKey = true

        for key in cacheKey {
            guard let child = node.children[key] else {
                walkedFullKey = false
                break
            }
            node = child
            if key >= 0 { realTokenCount += 1 }
            if let entryId = node.entryId,
               let entry = entries[entryId],
               entry.modelId == modelId {
                bestMatch = (entryId: entryId, realTokenCount: realTokenCount)
            }
        }

        if let match = bestMatch,
            var entry = entries[match.entryId] {
            entry.lastAccessAt = now
            entry.hitCount += 1
            entries[match.entryId] = entry
            removeEntryLocked(entry, countAsEviction: false)
            stats.totalHits += 1
            stats.totalPrefixHits += 1
            lock.unlock()

            // Dequantize if necessary before returning to caller
            let cacheToReturn = Self.dequantizeCache(entry.kvCache)

            return CacheLease(
                entryId: match.entryId,
                kvCache: cacheToReturn,
                matchedTokenCount: match.realTokenCount,
                isHit: true
            )
        }

        if walkedFullKey,
           let superLease = findSupersequenceMatchLocked(
               below: node,
               queryRealTokenCount: realTokenCount,
               modelId: modelId,
               now: now
           ) {
            lock.unlock()
            return superLease
        }

          if !walkedFullKey,
              realTokenCount > 0,
           let lcpLease = findLCPMatchLocked(
               below: node,
               sharedRealTokenCount: realTokenCount,
               queryRealTokenCount: queryRealTokenCount,
               modelId: modelId,
               now: now
           ) {
            lock.unlock()
            return lcpLease
        }

        stats.totalMisses += 1
        lock.unlock()
        return CacheLease(entryId: UUID(), kvCache: nil, matchedTokenCount: 0, isHit: false)
    }

    func store(
        entryId: UUID,
        kvCache: [KVCache],
        cacheKey: [Int],
        modelId: String
    ) {
        lock.lock()
        let now = nowProvider()
        pruneExpiredLocked(now: now)

        let normalizedCache = Self.normalizeCacheForStorage(kvCache)
        let bytesBeforeQuantization = estimateBytesProvider(normalizedCache)
        let cacheToStore: [KVCache]

        if quantizationConfig.enabled && cacheKey.filter({ $0 >= 0 }).count >= quantizationConfig.minTokens {
            cacheToStore = Self.quantizeCache(normalizedCache, config: quantizationConfig)
        } else {
            cacheToStore = normalizedCache
        }

        let isQuantized = Self.cacheContainsQuantizedLayers(cacheToStore)

        let estimatedBytes = estimateBytesProvider(cacheToStore)
        let bytesSaved = bytesBeforeQuantization - estimatedBytes

        // Update quantization stats if applicable
        if isQuantized && bytesSaved > 0 {
            stats.totalQuantizationBytesSaved += bytesSaved
        }

        var node = root
        for key in cacheKey {
            if node.children[key] == nil {
                node.children[key] = TrieNode()
            }
            node = node.children[key]!
        }

        if let oldId = node.entryId,
           let oldEntry = entries[oldId] {
            removeEntryLocked(oldEntry, countAsEviction: false)
        }

        node.entryId = entryId
        entries[entryId] = CacheEntry(
            id: entryId,
            modelId: modelId,
            kvCache: cacheToStore,
            tokenCount: cacheKey.filter { $0 >= 0 }.count,
            cacheKey: cacheKey,
            estimatedBytes: estimatedBytes,
            createdAt: now,
            lastAccessAt: now,
            hitCount: 0,
            isQuantized: isQuantized
        )
        currentMemoryBytes += estimatedBytes
        enforceBudgetLocked()
        lock.unlock()
    }

    func invalidateAll() {
        lock.lock()
        stats.totalEvictions += entries.count
        entries.removeAll()
        root = TrieNode()
        currentMemoryBytes = 0
        lock.unlock()
    }

    func reset() {
        lock.lock()
        root = TrieNode()
        entries.removeAll()
        currentMemoryBytes = 0
        stats = Stats()
        lock.unlock()
    }

    func snapshot() -> Snapshot {
        lock.lock()
        let now = nowProvider()
        pruneExpiredLocked(now: now)
        let orderedEntries = entries.values.sorted { lhs, rhs in
            if lhs.lastAccessAt != rhs.lastAccessAt {
                return lhs.lastAccessAt > rhs.lastAccessAt
            }
            return lhs.createdAt > rhs.createdAt
        }
        let hits = stats.totalHits
        let misses = stats.totalMisses
        let totalOps = hits + misses

        let snapshot = Snapshot(
            totalEntries: orderedEntries.count,
            totalCachedTokens: orderedEntries.reduce(0) { $0 + $1.tokenCount },
            estimatedBytes: currentMemoryBytes,
            memoryBudgetBytes: maxMemoryBytes,
            memoryUsagePercent: maxMemoryBytes > 0
                ? (Double(currentMemoryBytes) / Double(maxMemoryBytes)) * 100
                : 0,
            totalHits: hits,
            totalMisses: misses,
            totalEvictions: stats.totalEvictions,
            hitRate: totalOps > 0 ? (Double(hits) / Double(totalOps)) * 100 : 0,
            prefixHits: stats.totalPrefixHits,
            supersequenceHits: stats.totalSupersequenceHits,
            lcpHits: stats.totalLCPHits,
            quantizationBytesSaved: stats.totalQuantizationBytesSaved,
            quantizationEnabled: quantizationConfig.enabled,
            entries: orderedEntries.map {
                EntrySummary(
                    id: $0.id,
                    modelId: $0.modelId,
                    tokenCount: $0.tokenCount,
                    estimatedBytes: $0.estimatedBytes,
                    createdAt: $0.createdAt,
                    lastAccessAt: $0.lastAccessAt,
                    hitCount: $0.hitCount
                )
            }
        )
        lock.unlock()
        return snapshot
    }

    func debugTrieNodeCount() -> Int {
        lock.lock()
        let count = countNodes(root)
        lock.unlock()
        return count
    }

    private func pruneExpiredLocked(now: Date) {
        let expired = entries.values.filter {
            now.timeIntervalSince($0.lastAccessAt) > idleTTL
        }
        for entry in expired {
            removeEntryLocked(entry, countAsEviction: true)
        }
    }

    private func enforceBudgetLocked() {
        while currentMemoryBytes > maxMemoryBytes {
            guard let victim = entries.values.min(by: evictionOrder) else {
                break
            }
            removeEntryLocked(victim, countAsEviction: true)
        }
    }

    private func removeEntryLocked(_ entry: CacheEntry, countAsEviction: Bool) {
        guard entries[entry.id] != nil else { return }

        var node = root
        var path: [(parent: TrieNode, key: Int)] = []
        for key in entry.cacheKey {
            guard let child = node.children[key] else { break }
            path.append((parent: node, key: key))
            node = child
        }
        node.entryId = nil

        for (parent, key) in path.reversed() {
            guard let child = parent.children[key] else { continue }
            if child.children.isEmpty && child.entryId == nil {
                parent.children.removeValue(forKey: key)
            } else {
                break
            }
        }

        currentMemoryBytes = max(0, currentMemoryBytes - entry.estimatedBytes)
        entries.removeValue(forKey: entry.id)
        if countAsEviction {
            stats.totalEvictions += 1
        }
    }

    private func evictionOrder(lhs: CacheEntry, rhs: CacheEntry) -> Bool {
        if lhs.lastAccessAt != rhs.lastAccessAt {
            return lhs.lastAccessAt < rhs.lastAccessAt
        }
        if lhs.hitCount != rhs.hitCount {
            return lhs.hitCount < rhs.hitCount
        }
        return lhs.createdAt < rhs.createdAt
    }

    private func countNodes(_ node: TrieNode) -> Int {
        1 + node.children.values.reduce(0) { $0 + countNodes($1) }
    }

    private func findSupersequenceMatchLocked(
        below node: TrieNode,
        queryRealTokenCount: Int,
        modelId: String,
        now: Date
    ) -> CacheLease? {
        var queue: [TrieNode] = [node]
        var bestEntry: CacheEntry?

        while !queue.isEmpty {
            let current = queue.removeFirst()
            if let entryId = current.entryId,
               let entry = entries[entryId],
               entry.modelId == modelId,
               entry.tokenCount > queryRealTokenCount,
               entry.kvCache.allSatisfy({ $0.isTrimmable }) {
                if bestEntry == nil || entry.tokenCount < bestEntry!.tokenCount {
                    bestEntry = entry
                }
            }

            for child in current.children.values {
                queue.append(child)
            }
        }

        guard let entry = bestEntry,
              let trimmedCache = Self.trimCacheByOffset(entry.kvCache, trimBy: entry.tokenCount - queryRealTokenCount)
        else {
            return nil
        }

        var updatedEntry = entry
        updatedEntry.lastAccessAt = now
        updatedEntry.hitCount += 1
        entries[entry.id] = updatedEntry
        removeEntryLocked(updatedEntry, countAsEviction: false)
        stats.totalHits += 1
        stats.totalSupersequenceHits += 1

        // Dequantize if necessary before returning to caller
        let cacheToReturn = Self.dequantizeCache(trimmedCache)

        return CacheLease(
            entryId: updatedEntry.id,
            kvCache: cacheToReturn,
            matchedTokenCount: queryRealTokenCount,
            isHit: true
        )
    }

    private func findLCPMatchLocked(
        below node: TrieNode,
        sharedRealTokenCount: Int,
        queryRealTokenCount: Int,
        modelId: String,
        now: Date
    ) -> CacheLease? {
        guard sharedRealTokenCount >= Self.minimumLCPMatchTokens(for: queryRealTokenCount) else {
            return nil
        }

        var queue = Array(node.children.values)
        var bestEntry: CacheEntry?

        while !queue.isEmpty {
            let current = queue.removeFirst()
            if let entryId = current.entryId,
               let entry = entries[entryId],
               entry.modelId == modelId,
               entry.tokenCount > sharedRealTokenCount,
               entry.kvCache.allSatisfy({ $0.isTrimmable }) {
                if bestEntry == nil || entry.tokenCount < bestEntry!.tokenCount {
                    bestEntry = entry
                }
            }

            for child in current.children.values {
                queue.append(child)
            }
        }

        guard let entry = bestEntry,
              let trimmedCache = Self.trimCacheByOffset(entry.kvCache, trimBy: entry.tokenCount - sharedRealTokenCount)
        else {
            return nil
        }

        var updatedEntry = entry
        updatedEntry.lastAccessAt = now
        updatedEntry.hitCount += 1
        entries[entry.id] = updatedEntry
        removeEntryLocked(updatedEntry, countAsEviction: false)
        stats.totalHits += 1
        stats.totalLCPHits += 1

        // Dequantize if necessary before returning to caller
        let cacheToReturn = Self.dequantizeCache(trimmedCache)

        return CacheLease(
            entryId: updatedEntry.id,
            kvCache: cacheToReturn,
            matchedTokenCount: sharedRealTokenCount,
            isHit: true
        )
    }

    private static func trimCacheByOffset(_ cache: [KVCache], trimBy: Int) -> [KVCache]? {
        guard trimBy >= 0 else { return nil }
        guard trimBy > 0 else { return cache }

        for layer in cache {
            guard layer.isTrimmable else { return nil }
            let trimmed = layer.trim(trimBy)
            guard trimmed == trimBy else { return nil }
        }

        return cache
    }

    private static func minimumLCPMatchTokens(for queryRealTokenCount: Int) -> Int {
        guard queryRealTokenCount > 0 else { return .max }
        return max(2, (queryRealTokenCount + 1) / 2)
    }

    private static func computeMemoryBudget() -> Int {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return computeMemoryBudget(recommendedWorkingSetSize: nil)
        }
        return computeMemoryBudget(recommendedWorkingSetSize: Int(device.recommendedMaxWorkingSetSize))
    }

    static func computeMemoryBudget(recommendedWorkingSetSize: Int?) -> Int {
        guard let recommendedWorkingSetSize else {
            return 512 * 1024 * 1024
        }

        let budget = Int(Double(recommendedWorkingSetSize) * 0.20)
        return max(256 * 1024 * 1024, min(budget, 8 * 1024 * 1024 * 1024))
    }

    private static func estimateBytes(_ kvCache: [KVCache]) -> Int {
        var total = 0
        for layer in kvCache {
            for array in layer.state {
                total += array.nbytes
            }
        }
        return max(total, 1024)
    }

    // MARK: - Quantization Support

    /// Quantize a KV cache for compact storage (Phase 6 feature).
    /// Converts FP16 K/V tensors to a lower-bit representation.
    /// Returns the quantized cache or the original cache if quantization is skipped/unsupported.
    private static func quantizeCache(
        _ cache: [KVCache],
        config: QuantizationConfig
    ) -> [KVCache] {
        guard config.enabled else { return cache }

        return cache.map { layer in
            if layer is QuantizedKVCache {
                return layer
            }

            if let simpleLayer = layer as? KVCacheSimple {
                let quantized = simpleLayer.toQuantized(
                    groupSize: config.groupSize,
                    bits: config.bits
                )
                MLX.eval(quantized.state)
                return quantized
            }

            // Preserve non-standard cache types unchanged.
            return layer
        }
    }

    /// Dequantize a KV cache back to standard form before inference.
    /// If the cache was not quantized, returns it unchanged.
    private static func dequantizeCache(_ cache: [KVCache]) -> [KVCache] {
        cache.map { layer in
            if let quantizedLayer = layer as? QuantizedKVCache {
                let unquantized = quantizedLayer.toUnquantized()
                MLX.eval(unquantized.state)
                return unquantized
            }

            return layer
        }
    }

    private static func normalizeCacheForStorage(_ cache: [KVCache]) -> [KVCache] {
        cache.map { layer in
            if let quantizedLayer = layer as? QuantizedKVCache {
                let compact = QuantizedKVCache(
                    groupSize: quantizedLayer.groupSize,
                    bits: quantizedLayer.bits,
                    mode: quantizedLayer.mode
                )
                compact.state = quantizedLayer.state
                compact.offset = quantizedLayer.offset
                MLX.eval(compact.state)
                return compact
            }

            if let simpleLayer = layer as? KVCacheSimple {
                let compact = KVCacheSimple()
                compact.state = simpleLayer.state
                MLX.eval(compact.state)
                return compact
            }

            return layer
        }
    }

    private static func cacheContainsQuantizedLayers(_ cache: [KVCache]) -> Bool {
        cache.contains { $0 is QuantizedKVCache }
    }
}