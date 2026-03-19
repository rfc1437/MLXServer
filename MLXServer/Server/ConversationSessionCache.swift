import Foundation
import MLXLMCommon
import os

enum APISessionPhase: String, Sendable {
    case idle = "Idle"
    case sessionBuild = "Session Build"
    case prefilling = "Prefilling"
    case generating = "Generating"
}

/// Bounded cache of API chat sessions keyed by normalized conversation history.
/// The cache is internal-only and safe to sample from the monitor without involving MainActor.
final class ConversationSessionCache: @unchecked Sendable {
    static let shared = ConversationSessionCache()

    private let lock = OSAllocatedUnfairLock()

    private let maxEntries = 8
    private let maxCachedTokens = 256_000
    private let idleTTL: TimeInterval = 10 * 60

    private var entries: [UUID: Entry] = [:]
    private var totals = Totals()

    private init() {}

    struct Lease {
        let entryId: UUID
        let session: ChatSession?
        let reusedPromptTokens: Int
        let cacheHit: Bool
    }

    struct SessionSummary: Identifiable, Sendable {
        let id: UUID
        let modelId: String
        let phase: APISessionPhase
        let messageCount: Int
        let cachedTokenEstimate: Int
        let estimatedBytes: Int
        let inFlightRequests: Int
        let hitCount: Int
        let lastPromptTokens: Int
        let lastCompletionTokens: Int
        let lastReuseTokens: Int
        let createdAt: Date
        let lastAccessAt: Date
    }

    struct Snapshot: Sendable {
        let totalEntries: Int
        let warmEntries: Int
        let activeEntries: Int
        let generatingEntries: Int
        let estimatedBytes: Int
        let cachedTokenEstimate: Int
        let totalHits: Int
        let totalMisses: Int
        let totalEvictions: Int
        let totalReusePromptTokens: Int
        let totalRebuildPromptTokens: Int
        let sessions: [SessionSummary]
    }

    func checkoutSession(
        modelId: String,
        instructions: String,
        historySignatures: [UInt64],
        requestMessageCount: Int,
        estimatedPromptTokens: Int,
        estimatedBytes: Int
    ) -> Lease {
        lock.lock()
        let now = Date()
        pruneExpiredLocked(now: now)

        let instructionsHash = Self.stableHash(instructions)
        let match = entries
            .values
            .filter {
                $0.modelId == modelId
                    && $0.instructionsHash == instructionsHash
                    && $0.session != nil
                    && $0.inFlightRequests == 0
                    && Self.historyMatches(cached: $0.requestMessageSignatures, incoming: historySignatures)
            }
            .max { lhs, rhs in
                lhs.requestMessageSignatures.count < rhs.requestMessageSignatures.count
            }

        if let match {
            var entry = match
            entry.inFlightRequests += 1
            entry.lastAccessAt = now
            entry.phase = .prefilling
            entry.lastReuseTokens = max(entry.cachedTokenEstimate, estimatedPromptTokens)
            entry.hitCount += 1
            entries[entry.id] = entry
            totals.totalHits += 1
            totals.totalReusePromptTokens += entry.lastReuseTokens
            let lease = Lease(
                entryId: entry.id,
                session: entry.session,
                reusedPromptTokens: entry.lastReuseTokens,
                cacheHit: true
            )
            lock.unlock()
            return lease
        }

        let entryId = UUID()
        entries[entryId] = Entry(
            id: entryId,
            modelId: modelId,
            instructionsHash: instructionsHash,
            requestMessageSignatures: historySignatures,
            messageCount: requestMessageCount,
            cachedTokenEstimate: estimatedPromptTokens,
            estimatedBytes: estimatedBytes,
            createdAt: now,
            lastAccessAt: now,
            inFlightRequests: 1,
            hitCount: 0,
            phase: .sessionBuild,
            lastPromptTokens: 0,
            lastCompletionTokens: 0,
            lastReuseTokens: 0,
            session: nil
        )
        totals.totalMisses += 1
        totals.totalRebuildPromptTokens += estimatedPromptTokens
        lock.unlock()
        return Lease(entryId: entryId, session: nil, reusedPromptTokens: 0, cacheHit: false)
    }

    func markSessionBuild(entryId: UUID) {
        updatePhase(entryId: entryId, phase: .sessionBuild)
    }

    func markPrefilling(entryId: UUID) {
        updatePhase(entryId: entryId, phase: .prefilling)
    }

    func markGenerating(entryId: UUID, promptTokens: Int, completionTokens: Int) {
        lock.lock()
        if var entry = entries[entryId] {
            entry.phase = .generating
            entry.lastPromptTokens = promptTokens
            entry.lastCompletionTokens = completionTokens
            entry.cachedTokenEstimate = max(entry.cachedTokenEstimate, promptTokens + completionTokens)
            entry.lastAccessAt = Date()
            entries[entryId] = entry
        }
        lock.unlock()
    }

    func completeRequest(
        entryId: UUID,
        session: ChatSession,
        requestMessageSignatures: [UInt64],
        requestMessageCount: Int,
        estimatedPromptTokens: Int,
        estimatedBytes: Int,
        promptTokens: Int,
        completionTokens: Int
    ) {
        lock.lock()
        let now = Date()
        if var entry = entries[entryId] {
            entry.session = session
            entry.requestMessageSignatures = requestMessageSignatures
            entry.messageCount = requestMessageCount
            entry.cachedTokenEstimate = max(estimatedPromptTokens, promptTokens + completionTokens)
            entry.estimatedBytes = estimatedBytes
            entry.lastPromptTokens = promptTokens
            entry.lastCompletionTokens = completionTokens
            entry.lastAccessAt = now
            entry.inFlightRequests = max(0, entry.inFlightRequests - 1)
            entry.phase = .idle
            entries[entryId] = entry
            enforceBudgetLocked(now: now)
        }
        lock.unlock()
    }

    func abandonRequest(entryId: UUID) {
        lock.lock()
        if var entry = entries[entryId] {
            entry.inFlightRequests = max(0, entry.inFlightRequests - 1)
            if entry.session == nil && entry.inFlightRequests == 0 {
                entries.removeValue(forKey: entryId)
            } else {
                entry.phase = .idle
                entry.lastAccessAt = Date()
                entries[entryId] = entry
            }
        }
        lock.unlock()
    }

    func invalidateAll() {
        lock.lock()
        totals.totalEvictions += entries.count
        entries.removeAll()
        lock.unlock()
    }

    func reset() {
        lock.lock()
        entries.removeAll()
        totals = Totals()
        lock.unlock()
    }

    func snapshot() -> Snapshot {
        lock.lock()
        let now = Date()
        pruneExpiredLocked(now: now)
        let allEntries = Array(entries.values)
        let sessions = allEntries
            .sorted {
                if $0.inFlightRequests != $1.inFlightRequests {
                    return $0.inFlightRequests > $1.inFlightRequests
                }
                return $0.lastAccessAt > $1.lastAccessAt
            }
            .map {
                SessionSummary(
                    id: $0.id,
                    modelId: $0.modelId,
                    phase: $0.phase,
                    messageCount: $0.messageCount,
                    cachedTokenEstimate: $0.cachedTokenEstimate,
                    estimatedBytes: $0.estimatedBytes,
                    inFlightRequests: $0.inFlightRequests,
                    hitCount: $0.hitCount,
                    lastPromptTokens: $0.lastPromptTokens,
                    lastCompletionTokens: $0.lastCompletionTokens,
                    lastReuseTokens: $0.lastReuseTokens,
                    createdAt: $0.createdAt,
                    lastAccessAt: $0.lastAccessAt
                )
            }
        let snapshot = Snapshot(
            totalEntries: allEntries.count,
            warmEntries: allEntries.filter { $0.session != nil }.count,
            activeEntries: allEntries.filter { $0.inFlightRequests > 0 }.count,
            generatingEntries: allEntries.filter { $0.phase == .generating }.count,
            estimatedBytes: allEntries.reduce(0) { $0 + $1.estimatedBytes },
            cachedTokenEstimate: allEntries.reduce(0) { $0 + $1.cachedTokenEstimate },
            totalHits: totals.totalHits,
            totalMisses: totals.totalMisses,
            totalEvictions: totals.totalEvictions,
            totalReusePromptTokens: totals.totalReusePromptTokens,
            totalRebuildPromptTokens: totals.totalRebuildPromptTokens,
            sessions: sessions
        )
        lock.unlock()
        return snapshot
    }

    private func updatePhase(entryId: UUID, phase: APISessionPhase) {
        lock.lock()
        if var entry = entries[entryId] {
            entry.phase = phase
            entry.lastAccessAt = Date()
            entries[entryId] = entry
        }
        lock.unlock()
    }

    private func pruneExpiredLocked(now: Date) {
        let expired = entries.values.filter {
            $0.inFlightRequests == 0 && now.timeIntervalSince($0.lastAccessAt) > idleTTL
        }
        guard !expired.isEmpty else { return }
        for entry in expired {
            entries.removeValue(forKey: entry.id)
        }
        totals.totalEvictions += expired.count
    }

    private func enforceBudgetLocked(now: Date) {
        pruneExpiredLocked(now: now)

        func totalCachedTokens() -> Int {
            entries.values.reduce(0) { $0 + $1.cachedTokenEstimate }
        }

        while entries.count > maxEntries || totalCachedTokens() > maxCachedTokens {
            guard let victim = entries.values
                .filter({ $0.inFlightRequests == 0 })
                .sorted(by: evictionOrder)
                .first
            else {
                break
            }
            entries.removeValue(forKey: victim.id)
            totals.totalEvictions += 1
        }
    }

    private func evictionOrder(lhs: Entry, rhs: Entry) -> Bool {
        if lhs.lastAccessAt != rhs.lastAccessAt {
            return lhs.lastAccessAt < rhs.lastAccessAt
        }
        if lhs.cachedTokenEstimate != rhs.cachedTokenEstimate {
            return lhs.cachedTokenEstimate > rhs.cachedTokenEstimate
        }
        return lhs.createdAt < rhs.createdAt
    }

    private static func historyMatches(cached: [UInt64], incoming: [UInt64]) -> Bool {
        guard cached.count <= incoming.count,
              incoming.count <= cached.count + 1 else { return false }
        for (lhs, rhs) in zip(cached, incoming) where lhs != rhs {
            return false
        }
        return true
    }

    static func stableHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return hash
    }

    private struct Entry {
        let id: UUID
        let modelId: String
        let instructionsHash: UInt64
        var requestMessageSignatures: [UInt64]
        var messageCount: Int
        var cachedTokenEstimate: Int
        var estimatedBytes: Int
        let createdAt: Date
        var lastAccessAt: Date
        var inFlightRequests: Int
        var hitCount: Int
        var phase: APISessionPhase
        var lastPromptTokens: Int
        var lastCompletionTokens: Int
        var lastReuseTokens: Int
        var session: ChatSession?
    }

    private struct Totals {
        var totalHits: Int = 0
        var totalMisses: Int = 0
        var totalEvictions: Int = 0
        var totalReusePromptTokens: Int = 0
        var totalRebuildPromptTokens: Int = 0
    }
}