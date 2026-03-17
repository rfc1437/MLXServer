import Foundation
import os

// MARK: - Thread-safe live counters (written from any thread, no actor isolation)

/// Lock-protected counters that the generation loop writes to directly.
/// No MainActor requirement — the UI polls these via the 1Hz timer.
final class LiveCounters: @unchecked Sendable {
    static let shared = LiveCounters()

    private let lock = OSAllocatedUnfairLock()

    // Current request
    private var _activeRequests: Int = 0
    private var _promptTokens: Int = 0
    private var _generationTokens: Int = 0
    private var _tokensPerSecond: Double = 0
    private var _isPrefilling: Bool = false
    private var _isGenerating: Bool = false
    private var _contextMax: Int = 0

    // Cumulative
    private var _totalRequests: Int = 0
    private var _totalPromptTokens: Int = 0
    private var _totalGenerationTokens: Int = 0

    func requestStarted(contextLength: Int) {
        lock.lock()
        _activeRequests += 1
        _totalRequests += 1
        _isPrefilling = true
        _isGenerating = false
        _promptTokens = 0
        _generationTokens = 0
        _tokensPerSecond = 0
        _contextMax = contextLength
        lock.unlock()
    }

    func prefillCompleted(promptTokens: Int) {
        lock.lock()
        _isPrefilling = false
        _isGenerating = true
        _promptTokens = promptTokens
        _totalPromptTokens += promptTokens
        lock.unlock()
    }

    func tokenGenerated(tokensPerSecond: Double, totalGenerated: Int) {
        lock.lock()
        _generationTokens = totalGenerated
        _tokensPerSecond = tokensPerSecond
        lock.unlock()
    }

    func requestCompleted(generationTokens: Int) {
        lock.lock()
        _activeRequests = max(0, _activeRequests - 1)
        _totalGenerationTokens += generationTokens
        if _activeRequests == 0 {
            _isGenerating = false
            _isPrefilling = false
            _tokensPerSecond = 0
        }
        lock.unlock()
    }

    func reset() {
        lock.lock()
        _activeRequests = 0
        _promptTokens = 0
        _generationTokens = 0
        _tokensPerSecond = 0
        _isPrefilling = false
        _isGenerating = false
        _contextMax = 0
        _totalRequests = 0
        _totalPromptTokens = 0
        _totalGenerationTokens = 0
        lock.unlock()
    }

    /// Atomic snapshot for the UI timer.
    func snapshot() -> Snapshot {
        lock.lock()
        let s = Snapshot(
            activeRequests: _activeRequests,
            promptTokens: _promptTokens,
            generationTokens: _generationTokens,
            tokensPerSecond: _tokensPerSecond,
            isPrefilling: _isPrefilling,
            isGenerating: _isGenerating,
            contextMax: _contextMax,
            totalRequests: _totalRequests,
            totalPromptTokens: _totalPromptTokens,
            totalGenerationTokens: _totalGenerationTokens
        )
        lock.unlock()
        return s
    }

    struct Snapshot {
        let activeRequests: Int
        let promptTokens: Int
        let generationTokens: Int
        let tokensPerSecond: Double
        let isPrefilling: Bool
        let isGenerating: Bool
        let contextMax: Int
        let totalRequests: Int
        let totalPromptTokens: Int
        let totalGenerationTokens: Int
    }
}

// MARK: - Observable stats for the UI (polls LiveCounters at 1Hz)

@Observable
@MainActor
final class InferenceStats {
    // MARK: - Current request state (refreshed from LiveCounters)

    var activeRequests: Int = 0
    var currentPromptTokens: Int = 0
    var currentGenerationTokens: Int = 0
    var isGenerating: Bool = false
    var isPrefilling: Bool = false
    var currentTokensPerSecond: Double = 0
    var contextUsed: Int = 0
    var contextMax: Int = 0

    // MARK: - Cumulative counters

    var totalRequests: Int = 0
    var totalPromptTokens: Int = 0
    var totalGenerationTokens: Int = 0

    // MARK: - Time series data (ring buffers for charts)

    struct DataPoint: Identifiable {
        let id = UUID()
        let timestamp: Date
        let value: Double
    }

    private(set) var tokenRateHistory: [DataPoint] = []
    private(set) var promptTokenHistory: [DataPoint] = []
    private(set) var generationTokenHistory: [DataPoint] = []

    private static let maxHistoryPoints = 120 // ~2 minutes at 1Hz

    // Periodic sampling
    private var sampleTimer: Timer?
    private var lastGenerationTokenCount: Int = 0
    private var lastPromptTokenCount: Int = 0

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

        activeRequests = snap.activeRequests
        currentPromptTokens = snap.promptTokens
        currentGenerationTokens = snap.generationTokens
        currentTokensPerSecond = snap.tokensPerSecond
        isPrefilling = snap.isPrefilling
        isGenerating = snap.isGenerating
        contextMax = snap.contextMax
        contextUsed = snap.promptTokens + snap.generationTokens
        totalRequests = snap.totalRequests
        totalPromptTokens = snap.totalPromptTokens
        totalGenerationTokens = snap.totalGenerationTokens

        let now = Date.now
        let genDelta = snap.totalGenerationTokens - lastGenerationTokenCount
        let promptDelta = snap.totalPromptTokens - lastPromptTokenCount
        lastGenerationTokenCount = snap.totalGenerationTokens
        lastPromptTokenCount = snap.totalPromptTokens

        tokenRateHistory.append(DataPoint(timestamp: now, value: snap.tokensPerSecond))
        generationTokenHistory.append(DataPoint(timestamp: now, value: Double(genDelta)))
        promptTokenHistory.append(DataPoint(timestamp: now, value: Double(promptDelta)))

        if tokenRateHistory.count > Self.maxHistoryPoints {
            tokenRateHistory.removeFirst(tokenRateHistory.count - Self.maxHistoryPoints)
        }
        if generationTokenHistory.count > Self.maxHistoryPoints {
            generationTokenHistory.removeFirst(generationTokenHistory.count - Self.maxHistoryPoints)
        }
        if promptTokenHistory.count > Self.maxHistoryPoints {
            promptTokenHistory.removeFirst(promptTokenHistory.count - Self.maxHistoryPoints)
        }
    }

    func reset() {
        LiveCounters.shared.reset()
        activeRequests = 0
        currentPromptTokens = 0
        currentGenerationTokens = 0
        isGenerating = false
        isPrefilling = false
        currentTokensPerSecond = 0
        contextUsed = 0
        contextMax = 0
        totalRequests = 0
        totalPromptTokens = 0
        totalGenerationTokens = 0
        tokenRateHistory.removeAll()
        promptTokenHistory.removeAll()
        generationTokenHistory.removeAll()
        lastGenerationTokenCount = 0
        lastPromptTokenCount = 0
    }
}
