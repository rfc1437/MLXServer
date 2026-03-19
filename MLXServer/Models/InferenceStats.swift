import Foundation
import os

// MARK: - Thread-safe live counters (written from any thread, no actor isolation)

/// Lock-protected counters that the generation loop writes to directly.
/// No MainActor requirement — the UI polls these via the 1Hz timer.
final class LiveCounters: @unchecked Sendable {
    static let shared = LiveCounters()

    private let lock = OSAllocatedUnfairLock()
    private var requestPhases: [String: RequestState] = [:]

    // Current request
    private var _activeRequests: Int = 0
    private var _preparingRequests: Int = 0
    private var _sessionBuildRequests: Int = 0
    private var _prefillRequests: Int = 0
    private var _generatingRequests: Int = 0
    private var _promptTokens: Int = 0
    private var _generationTokens: Int = 0
    private var _tokensPerSecond: Double = 0
    private var _isPrefilling: Bool = false
    private var _isGenerating: Bool = false
    private var _contextMax: Int = 0
    private var _currentPhaseElapsed: TimeInterval = 0

    // Cumulative
    private var _totalRequests: Int = 0
    private var _totalPromptTokens: Int = 0
    private var _totalGenerationTokens: Int = 0
    private var _totalPreparingDuration: TimeInterval = 0
    private var _totalSessionBuildDuration: TimeInterval = 0
    private var _totalPrefillDuration: TimeInterval = 0
    private var _totalGenerationDuration: TimeInterval = 0

    func requestStarted(requestId: String, contextLength: Int) {
        let now = Date()
        lock.lock()
        _activeRequests += 1
        _preparingRequests += 1
        _totalRequests += 1
        _isPrefilling = true
        _isGenerating = false
        _promptTokens = 0
        _generationTokens = 0
        _tokensPerSecond = 0
        _contextMax = contextLength
        requestPhases[requestId] = RequestState(phase: .preparing, phaseStartedAt: now)
        refreshCurrentPhaseElapsed(now: now)
        lock.unlock()
    }

    func requestPhaseChanged(requestId: String, phase: RequestPhase) {
        let now = Date()
        lock.lock()
        if let current = requestPhases[requestId] {
            decrementCount(for: current.phase)
            accumulateDuration(for: current.phase, elapsed: now.timeIntervalSince(current.phaseStartedAt))
        }
        incrementCount(for: phase)
        requestPhases[requestId] = RequestState(phase: phase, phaseStartedAt: now)
        _isPrefilling = _prefillRequests > 0 || _sessionBuildRequests > 0 || _preparingRequests > 0
        _isGenerating = _generatingRequests > 0
        refreshCurrentPhaseElapsed(now: now)
        lock.unlock()
    }

    func prefillCompleted(requestId: String, promptTokens: Int) {
        let now = Date()
        lock.lock()
        if let current = requestPhases[requestId] {
            decrementCount(for: current.phase)
            accumulateDuration(for: current.phase, elapsed: now.timeIntervalSince(current.phaseStartedAt))
        }
        incrementCount(for: .generating)
        requestPhases[requestId] = RequestState(phase: .generating, phaseStartedAt: now)
        _promptTokens = promptTokens
        _totalPromptTokens += promptTokens
        _isPrefilling = _prefillRequests > 0 || _sessionBuildRequests > 0 || _preparingRequests > 0
        _isGenerating = _generatingRequests > 0
        refreshCurrentPhaseElapsed(now: now)
        lock.unlock()
    }

    func tokenGenerated(tokensPerSecond: Double, totalGenerated: Int) {
        lock.lock()
        _generationTokens = totalGenerated
        _tokensPerSecond = tokensPerSecond
        lock.unlock()
    }

    func requestCompleted(requestId: String, generationTokens: Int) {
        let now = Date()
        lock.lock()
        if let current = requestPhases.removeValue(forKey: requestId) {
            decrementCount(for: current.phase)
            accumulateDuration(for: current.phase, elapsed: now.timeIntervalSince(current.phaseStartedAt))
        }
        _activeRequests = max(0, _activeRequests - 1)
        _totalGenerationTokens += generationTokens
        if _activeRequests == 0 {
            _isGenerating = false
            _isPrefilling = false
            _tokensPerSecond = 0
        } else {
            _isPrefilling = _prefillRequests > 0 || _sessionBuildRequests > 0 || _preparingRequests > 0
            _isGenerating = _generatingRequests > 0
        }
        refreshCurrentPhaseElapsed(now: now)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        requestPhases.removeAll()
        _activeRequests = 0
        _preparingRequests = 0
        _sessionBuildRequests = 0
        _prefillRequests = 0
        _generatingRequests = 0
        _promptTokens = 0
        _generationTokens = 0
        _tokensPerSecond = 0
        _isPrefilling = false
        _isGenerating = false
        _contextMax = 0
        _currentPhaseElapsed = 0
        _totalRequests = 0
        _totalPromptTokens = 0
        _totalGenerationTokens = 0
        _totalPreparingDuration = 0
        _totalSessionBuildDuration = 0
        _totalPrefillDuration = 0
        _totalGenerationDuration = 0
        lock.unlock()
    }

    /// Atomic snapshot for the UI timer.
    func snapshot() -> Snapshot {
        let now = Date()
        lock.lock()
        refreshCurrentPhaseElapsed(now: now)
        let s = Snapshot(
            activeRequests: _activeRequests,
            preparingRequests: _preparingRequests,
            sessionBuildRequests: _sessionBuildRequests,
            prefillRequests: _prefillRequests,
            generatingRequests: _generatingRequests,
            promptTokens: _promptTokens,
            generationTokens: _generationTokens,
            tokensPerSecond: _tokensPerSecond,
            isPrefilling: _isPrefilling,
            isGenerating: _isGenerating,
            contextMax: _contextMax,
            currentPhaseElapsed: _currentPhaseElapsed,
            totalRequests: _totalRequests,
            totalPromptTokens: _totalPromptTokens,
            totalGenerationTokens: _totalGenerationTokens,
            totalPreparingDuration: _totalPreparingDuration,
            totalSessionBuildDuration: _totalSessionBuildDuration,
            totalPrefillDuration: _totalPrefillDuration,
            totalGenerationDuration: _totalGenerationDuration
        )
        lock.unlock()
        return s
    }

    struct Snapshot {
        let activeRequests: Int
        let preparingRequests: Int
        let sessionBuildRequests: Int
        let prefillRequests: Int
        let generatingRequests: Int
        let promptTokens: Int
        let generationTokens: Int
        let tokensPerSecond: Double
        let isPrefilling: Bool
        let isGenerating: Bool
        let contextMax: Int
        let currentPhaseElapsed: TimeInterval
        let totalRequests: Int
        let totalPromptTokens: Int
        let totalGenerationTokens: Int
        let totalPreparingDuration: TimeInterval
        let totalSessionBuildDuration: TimeInterval
        let totalPrefillDuration: TimeInterval
        let totalGenerationDuration: TimeInterval
    }

    private func incrementCount(for phase: RequestPhase) {
        switch phase {
        case .preparing:
            _preparingRequests += 1
        case .sessionBuild:
            _sessionBuildRequests += 1
        case .prefilling:
            _prefillRequests += 1
        case .generating:
            _generatingRequests += 1
        }
    }

    private func decrementCount(for phase: RequestPhase) {
        switch phase {
        case .preparing:
            _preparingRequests = max(0, _preparingRequests - 1)
        case .sessionBuild:
            _sessionBuildRequests = max(0, _sessionBuildRequests - 1)
        case .prefilling:
            _prefillRequests = max(0, _prefillRequests - 1)
        case .generating:
            _generatingRequests = max(0, _generatingRequests - 1)
        }
    }

    private func accumulateDuration(for phase: RequestPhase, elapsed: TimeInterval) {
        switch phase {
        case .preparing:
            _totalPreparingDuration += elapsed
        case .sessionBuild:
            _totalSessionBuildDuration += elapsed
        case .prefilling:
            _totalPrefillDuration += elapsed
        case .generating:
            _totalGenerationDuration += elapsed
        }
    }

    private func refreshCurrentPhaseElapsed(now: Date) {
        _currentPhaseElapsed = requestPhases.values.map { now.timeIntervalSince($0.phaseStartedAt) }.max() ?? 0
    }

    private struct RequestState {
        var phase: RequestPhase
        var phaseStartedAt: Date
    }

    enum RequestPhase {
        case preparing
        case sessionBuild
        case prefilling
        case generating
    }
}

// MARK: - Observable stats for the UI (polls LiveCounters at 1Hz)

@Observable
@MainActor
final class InferenceStats {
    // MARK: - Current request state (refreshed from LiveCounters)

    var activeRequests: Int = 0
    var preparingRequests: Int = 0
    var sessionBuildRequests: Int = 0
    var prefillingRequests: Int = 0
    var generatingRequests: Int = 0
    var currentPromptTokens: Int = 0
    var currentGenerationTokens: Int = 0
    var isGenerating: Bool = false
    var isPrefilling: Bool = false
    var currentTokensPerSecond: Double = 0
    var contextUsed: Int = 0
    var contextMax: Int = 0
    var currentPhaseElapsed: TimeInterval = 0

    // MARK: - Cumulative counters

    var totalRequests: Int = 0
    var totalPromptTokens: Int = 0
    var totalGenerationTokens: Int = 0
    var totalCacheHits: Int = 0
    var totalCacheMisses: Int = 0
    var totalCacheEvictions: Int = 0
    var totalCacheReusePromptTokens: Int = 0
    var totalCacheRebuildPromptTokens: Int = 0
    var totalPreparingDuration: TimeInterval = 0
    var totalSessionBuildDuration: TimeInterval = 0
    var totalPrefillDuration: TimeInterval = 0
    var totalGenerationDuration: TimeInterval = 0

    // MARK: - Cache state

    var cacheEntryCount: Int = 0
    var warmCacheEntryCount: Int = 0
    var activeCacheEntryCount: Int = 0
    var generatingCacheEntryCount: Int = 0
    var cacheEstimatedBytes: Int = 0
    var cacheEstimatedTokens: Int = 0
    var cachedSessions: [ConversationSessionCache.SessionSummary] = []

    // MARK: - Time series data (ring buffers for charts)

    struct DataPoint: Identifiable {
        let id = UUID()
        let timestamp: Date
        let value: Double
    }

    private(set) var tokenRateHistory: [DataPoint] = []
    private(set) var promptTokenHistory: [DataPoint] = []
    private(set) var generationTokenHistory: [DataPoint] = []
    private(set) var cacheEntryHistory: [DataPoint] = []
    private(set) var activeSessionHistory: [DataPoint] = []
    private(set) var cacheFootprintHistory: [DataPoint] = []
    private(set) var cacheReuseHistory: [DataPoint] = []
    private(set) var cacheRebuildHistory: [DataPoint] = []
    private(set) var currentPhaseElapsedHistory: [DataPoint] = []
    private(set) var prefillDurationHistory: [DataPoint] = []
    private(set) var sessionBuildDurationHistory: [DataPoint] = []

    private static let maxHistoryPoints = 120 // ~2 minutes at 1Hz

    // Periodic sampling
    private var sampleTimer: Timer?
    private var lastGenerationTokenCount: Int = 0
    private var lastPromptTokenCount: Int = 0
    private var lastCacheReuseTokenCount: Int = 0
    private var lastCacheRebuildTokenCount: Int = 0
    private var lastPrefillDuration: TimeInterval = 0
    private var lastSessionBuildDuration: TimeInterval = 0

    func startSampling() {
        guard sampleTimer == nil else { return }
        sampleTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.recordSample()
            }
        }
    }

    func stopSampling() {
        sampleTimer?.invalidate()
        sampleTimer = nil
    }

    private func recordSample() {
        // Pull live values from the thread-safe counters
        let snap = LiveCounters.shared.snapshot()
        let cache = ConversationSessionCache.shared.snapshot()

        activeRequests = snap.activeRequests
        preparingRequests = snap.preparingRequests
        sessionBuildRequests = snap.sessionBuildRequests
        prefillingRequests = snap.prefillRequests
        generatingRequests = snap.generatingRequests
        currentPromptTokens = snap.promptTokens
        currentGenerationTokens = snap.generationTokens
        currentTokensPerSecond = snap.tokensPerSecond
        isPrefilling = snap.isPrefilling
        isGenerating = snap.isGenerating
        contextMax = snap.contextMax
        contextUsed = snap.promptTokens + snap.generationTokens
        currentPhaseElapsed = snap.currentPhaseElapsed
        totalRequests = snap.totalRequests
        totalPromptTokens = snap.totalPromptTokens
        totalGenerationTokens = snap.totalGenerationTokens
        totalPreparingDuration = snap.totalPreparingDuration
        totalSessionBuildDuration = snap.totalSessionBuildDuration
        totalPrefillDuration = snap.totalPrefillDuration
        totalGenerationDuration = snap.totalGenerationDuration
        totalCacheHits = cache.totalHits
        totalCacheMisses = cache.totalMisses
        totalCacheEvictions = cache.totalEvictions
        totalCacheReusePromptTokens = cache.totalReusePromptTokens
        totalCacheRebuildPromptTokens = cache.totalRebuildPromptTokens
        cacheEntryCount = cache.totalEntries
        warmCacheEntryCount = cache.warmEntries
        activeCacheEntryCount = cache.activeEntries
        generatingCacheEntryCount = cache.generatingEntries
        cacheEstimatedBytes = cache.estimatedBytes
        cacheEstimatedTokens = cache.cachedTokenEstimate
        cachedSessions = cache.sessions

        let now = Date.now
        let genDelta = snap.totalGenerationTokens - lastGenerationTokenCount
        let promptDelta = snap.totalPromptTokens - lastPromptTokenCount
        let cacheReuseDelta = cache.totalReusePromptTokens - lastCacheReuseTokenCount
        let cacheRebuildDelta = cache.totalRebuildPromptTokens - lastCacheRebuildTokenCount
        let prefillDurationDelta = snap.totalPrefillDuration - lastPrefillDuration
        let sessionBuildDurationDelta = snap.totalSessionBuildDuration - lastSessionBuildDuration
        lastGenerationTokenCount = snap.totalGenerationTokens
        lastPromptTokenCount = snap.totalPromptTokens
        lastCacheReuseTokenCount = cache.totalReusePromptTokens
        lastCacheRebuildTokenCount = cache.totalRebuildPromptTokens
        lastPrefillDuration = snap.totalPrefillDuration
        lastSessionBuildDuration = snap.totalSessionBuildDuration

        tokenRateHistory.append(DataPoint(timestamp: now, value: snap.tokensPerSecond))
        generationTokenHistory.append(DataPoint(timestamp: now, value: Double(genDelta)))
        promptTokenHistory.append(DataPoint(timestamp: now, value: Double(promptDelta)))
        cacheEntryHistory.append(DataPoint(timestamp: now, value: Double(cache.totalEntries)))
        activeSessionHistory.append(DataPoint(timestamp: now, value: Double(cache.activeEntries)))
        cacheFootprintHistory.append(DataPoint(timestamp: now, value: Double(cache.estimatedBytes)))
        cacheReuseHistory.append(DataPoint(timestamp: now, value: Double(cacheReuseDelta)))
        cacheRebuildHistory.append(DataPoint(timestamp: now, value: Double(cacheRebuildDelta)))
        currentPhaseElapsedHistory.append(DataPoint(timestamp: now, value: snap.currentPhaseElapsed))
        prefillDurationHistory.append(DataPoint(timestamp: now, value: prefillDurationDelta))
        sessionBuildDurationHistory.append(DataPoint(timestamp: now, value: sessionBuildDurationDelta))

        if tokenRateHistory.count > Self.maxHistoryPoints {
            tokenRateHistory.removeFirst(tokenRateHistory.count - Self.maxHistoryPoints)
        }
        if generationTokenHistory.count > Self.maxHistoryPoints {
            generationTokenHistory.removeFirst(generationTokenHistory.count - Self.maxHistoryPoints)
        }
        if promptTokenHistory.count > Self.maxHistoryPoints {
            promptTokenHistory.removeFirst(promptTokenHistory.count - Self.maxHistoryPoints)
        }
        if cacheEntryHistory.count > Self.maxHistoryPoints {
            cacheEntryHistory.removeFirst(cacheEntryHistory.count - Self.maxHistoryPoints)
        }
        if activeSessionHistory.count > Self.maxHistoryPoints {
            activeSessionHistory.removeFirst(activeSessionHistory.count - Self.maxHistoryPoints)
        }
        if cacheFootprintHistory.count > Self.maxHistoryPoints {
            cacheFootprintHistory.removeFirst(cacheFootprintHistory.count - Self.maxHistoryPoints)
        }
        if cacheReuseHistory.count > Self.maxHistoryPoints {
            cacheReuseHistory.removeFirst(cacheReuseHistory.count - Self.maxHistoryPoints)
        }
        if cacheRebuildHistory.count > Self.maxHistoryPoints {
            cacheRebuildHistory.removeFirst(cacheRebuildHistory.count - Self.maxHistoryPoints)
        }
        if currentPhaseElapsedHistory.count > Self.maxHistoryPoints {
            currentPhaseElapsedHistory.removeFirst(currentPhaseElapsedHistory.count - Self.maxHistoryPoints)
        }
        if prefillDurationHistory.count > Self.maxHistoryPoints {
            prefillDurationHistory.removeFirst(prefillDurationHistory.count - Self.maxHistoryPoints)
        }
        if sessionBuildDurationHistory.count > Self.maxHistoryPoints {
            sessionBuildDurationHistory.removeFirst(sessionBuildDurationHistory.count - Self.maxHistoryPoints)
        }
    }

    func reset() {
        LiveCounters.shared.reset()
        ConversationSessionCache.shared.reset()
        activeRequests = 0
        preparingRequests = 0
        sessionBuildRequests = 0
        prefillingRequests = 0
        generatingRequests = 0
        currentPromptTokens = 0
        currentGenerationTokens = 0
        isGenerating = false
        isPrefilling = false
        currentTokensPerSecond = 0
        contextUsed = 0
        contextMax = 0
        currentPhaseElapsed = 0
        totalRequests = 0
        totalPromptTokens = 0
        totalGenerationTokens = 0
        totalPreparingDuration = 0
        totalSessionBuildDuration = 0
        totalPrefillDuration = 0
        totalGenerationDuration = 0
        totalCacheHits = 0
        totalCacheMisses = 0
        totalCacheEvictions = 0
        totalCacheReusePromptTokens = 0
        totalCacheRebuildPromptTokens = 0
        cacheEntryCount = 0
        warmCacheEntryCount = 0
        activeCacheEntryCount = 0
        generatingCacheEntryCount = 0
        cacheEstimatedBytes = 0
        cacheEstimatedTokens = 0
        cachedSessions.removeAll()
        tokenRateHistory.removeAll()
        promptTokenHistory.removeAll()
        generationTokenHistory.removeAll()
        cacheEntryHistory.removeAll()
        activeSessionHistory.removeAll()
        cacheFootprintHistory.removeAll()
        cacheReuseHistory.removeAll()
        cacheRebuildHistory.removeAll()
        currentPhaseElapsedHistory.removeAll()
        prefillDurationHistory.removeAll()
        sessionBuildDurationHistory.removeAll()
        lastGenerationTokenCount = 0
        lastPromptTokenCount = 0
        lastCacheReuseTokenCount = 0
        lastCacheRebuildTokenCount = 0
        lastPrefillDuration = 0
        lastSessionBuildDuration = 0
    }
}
