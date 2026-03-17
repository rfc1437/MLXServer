import Foundation

/// Lightweight stats collector for inference activity visualization.
/// All mutations happen on @MainActor to avoid locks.
@Observable
@MainActor
final class InferenceStats {
    // MARK: - Current request state

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
    private var lastSampleTime: Date = .now

    func startSampling() {
        guard sampleTimer == nil else { return }
        lastSampleTime = .now
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
        let now = Date.now

        // Token rate: tokens generated since last sample
        let genDelta = totalGenerationTokens - lastGenerationTokenCount
        let promptDelta = totalPromptTokens - lastPromptTokenCount
        lastGenerationTokenCount = totalGenerationTokens
        lastPromptTokenCount = totalPromptTokens

        tokenRateHistory.append(DataPoint(timestamp: now, value: currentTokensPerSecond))
        generationTokenHistory.append(DataPoint(timestamp: now, value: Double(genDelta)))
        promptTokenHistory.append(DataPoint(timestamp: now, value: Double(promptDelta)))

        // Trim to ring buffer size
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

    // MARK: - Event recording (called from APIServer)

    func requestStarted(contextLength: Int) {
        activeRequests += 1
        totalRequests += 1
        isPrefilling = true
        isGenerating = false
        currentPromptTokens = 0
        currentGenerationTokens = 0
        currentTokensPerSecond = 0
        contextMax = contextLength
        contextUsed = 0
    }

    func prefillCompleted(promptTokens: Int) {
        isPrefilling = false
        isGenerating = true
        currentPromptTokens = promptTokens
        totalPromptTokens += promptTokens
        contextUsed = promptTokens
    }

    func tokenGenerated(tokensPerSecond: Double, totalGenerated: Int) {
        currentGenerationTokens = totalGenerated
        currentTokensPerSecond = tokensPerSecond
        contextUsed = currentPromptTokens + totalGenerated
    }

    func requestCompleted(promptTokens: Int, generationTokens: Int) {
        activeRequests = max(0, activeRequests - 1)
        totalGenerationTokens += generationTokens
        if activeRequests == 0 {
            isGenerating = false
            isPrefilling = false
            currentTokensPerSecond = 0
        }
    }

    func reset() {
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
