import Foundation
import Metal
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
    }

    private struct Stats {
        var totalHits: Int = 0
        var totalMisses: Int = 0
        var totalEvictions: Int = 0
        var totalPrefixHits: Int = 0
        var totalSupersequenceHits: Int = 0
        var totalLCPHits: Int = 0
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

    private init() {
        self.maxMemoryBytes = Self.computeMemoryBudget()
        self.idleTTL = 30 * 60
        self.estimateBytesProvider = Self.estimateBytes
        self.nowProvider = Date.init
    }

    init(
        memoryBudgetBytes: Int,
        idleTTL: TimeInterval = 30 * 60,
        estimateBytesProvider: @escaping ([KVCache]) -> Int = TokenPrefixCache.estimateBytes,
        nowProvider: @escaping () -> Date = Date.init
    ) {
        self.maxMemoryBytes = memoryBudgetBytes
        self.idleTTL = idleTTL
        self.estimateBytesProvider = estimateBytesProvider
        self.nowProvider = nowProvider
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
            removeEntryLocked(entry)
            stats.totalHits += 1
            stats.totalPrefixHits += 1
            lock.unlock()

            return CacheLease(
                entryId: match.entryId,
                kvCache: entry.kvCache,
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

        if realTokenCount > 0,
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

        let estimatedBytes = estimateBytesProvider(kvCache)
        var node = root
        for key in cacheKey {
            if node.children[key] == nil {
                node.children[key] = TrieNode()
            }
            node = node.children[key]!
        }

        if let oldId = node.entryId,
           let oldEntry = entries[oldId] {
            removeEntryLocked(oldEntry)
        }

        node.entryId = entryId
        entries[entryId] = CacheEntry(
            id: entryId,
            modelId: modelId,
            kvCache: kvCache,
            tokenCount: cacheKey.filter { $0 >= 0 }.count,
            cacheKey: cacheKey,
            estimatedBytes: estimatedBytes,
            createdAt: now,
            lastAccessAt: now,
            hitCount: 0
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
            removeEntryLocked(entry)
        }
    }

    private func enforceBudgetLocked() {
        while currentMemoryBytes > maxMemoryBytes {
            guard let victim = entries.values.min(by: evictionOrder) else {
                break
            }
            removeEntryLocked(victim)
        }
    }

    private func removeEntryLocked(_ entry: CacheEntry) {
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
        stats.totalEvictions += 1
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
        removeEntryLocked(updatedEntry)
        stats.totalHits += 1
        stats.totalSupersequenceHits += 1

        return CacheLease(
            entryId: updatedEntry.id,
            kvCache: trimmedCache,
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
        removeEntryLocked(updatedEntry)
        stats.totalHits += 1
        stats.totalLCPHits += 1

        return CacheLease(
            entryId: updatedEntry.id,
            kvCache: trimmedCache,
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
}