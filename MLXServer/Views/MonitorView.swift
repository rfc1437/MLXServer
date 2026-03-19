import Charts
import MLX
import SwiftUI

/// Real-time inference monitoring dashboard, shown in place of the chat UI.
struct MonitorView: View {
    let stats: InferenceStats
    @Environment(ModelManager.self) private var modelManager
    private let chartColumns = [GridItem(.flexible(minimum: 260), spacing: 16), GridItem(.flexible(minimum: 260), spacing: 16)]
    private let cardColumns = [GridItem(.flexible(minimum: 180), spacing: 16), GridItem(.flexible(minimum: 180), spacing: 16)]

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                liveStatusSection

                LazyVGrid(columns: chartColumns, alignment: .leading, spacing: 16) {
                    tokenRateChart
                    tokenThroughputChart
                    phaseActivityChart
                    cacheReuseChart
                    cacheFootprintChart
                    cacheSessionChart
                }

                LazyVGrid(columns: cardColumns, alignment: .leading, spacing: 16) {
                    contextGauge
                    gpuMemoryGauge
                    requestsCard
                    cacheCard
                }

                cumulativeSection
                sessionSection
            }
            .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }

    // MARK: - Live Status

    @ViewBuilder
    private var liveStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 16) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(activityColor)
                        .frame(width: 10, height: 10)
                        .overlay {
                            if stats.activeRequests > 0 {
                                Circle()
                                    .stroke(activityColor.opacity(0.5), lineWidth: 2)
                                    .scaleEffect(1.8)
                                    .opacity(0.6)
                            }
                        }

                    Text(activityLabel)
                        .font(.headline)
                }

                Spacer()

                if stats.isGenerating {
                    Text(String(format: "%.1f tok/s", stats.currentTokensPerSecond))
                        .font(.title2.monospacedDigit().bold())
                        .foregroundStyle(.green)
                }

                if stats.currentPromptTokens > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.circle.fill")
                            .foregroundStyle(.blue)
                        Text("\(stats.currentPromptTokens)")
                            .monospacedDigit()
                        Image(systemName: "arrow.up.circle.fill")
                            .foregroundStyle(.orange)
                        Text("\(stats.currentGenerationTokens)")
                            .monospacedDigit()
                    }
                    .font(.callout)
                }
            }

            HStack(spacing: 8) {
                phaseChip(title: "Preparing", count: stats.preparingRequests, color: .secondary)
                phaseChip(title: "Session Build", count: stats.sessionBuildRequests, color: .purple)
                phaseChip(title: "Prefill", count: stats.prefillingRequests, color: .blue)
                phaseChip(title: "Generating", count: stats.generatingRequests, color: .green)
                phaseChip(title: "Cache Active", count: stats.activeCacheEntryCount, color: .orange)
                if stats.activeRequests > 0 {
                    phaseChip(title: phaseAgeLabel, count: Int(stats.currentPhaseElapsed.rounded()), color: activityColor)
                }
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private var activityColor: Color {
        if stats.isGenerating { return .green }
        if stats.prefillingRequests > 0 { return .blue }
        if stats.sessionBuildRequests > 0 { return .purple }
        if stats.preparingRequests > 0 { return .orange }
        if stats.activeRequests > 0 { return .orange }
        return .secondary
    }

    private var activityLabel: String {
        if stats.isGenerating { return "Generating" }
        if stats.prefillingRequests > 0 { return "Prefilling" }
        if stats.sessionBuildRequests > 0 { return "Building Sessions" }
        if stats.preparingRequests > 0 { return "Preparing Requests" }
        if stats.activeRequests > 0 { return "Processing" }
        return "Idle"
    }

    // MARK: - Token Rate Chart

    @ViewBuilder
    private var tokenRateChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Generation Speed (tok/s)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart(stats.tokenRateHistory) { point in
                LineMark(
                    x: .value("Time", point.timestamp),
                    y: .value("tok/s", point.value)
                )
                .foregroundStyle(.green)
                .interpolationMethod(.monotone)

                AreaMark(
                    x: .value("Time", point.timestamp),
                    y: .value("tok/s", point.value)
                )
                .foregroundStyle(.green.opacity(0.1))
                .interpolationMethod(.monotone)
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.0f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .chartYScale(domain: 0...(maxTokenRate + 1))
            .frame(height: 150)
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var phaseActivityChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Phase Activity")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart {
                ForEach(stats.currentPhaseElapsedHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Active s", point.value)
                    )
                    .foregroundStyle(activityColor)
                    .interpolationMethod(.monotone)
                }
                ForEach(stats.prefillDurationHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Prefill done", point.value)
                    )
                    .foregroundStyle(.blue.opacity(0.45))
                }
                ForEach(stats.sessionBuildDurationHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Build done", point.value)
                    )
                    .foregroundStyle(.purple.opacity(0.45))
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.0f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .frame(height: 150)

            HStack(spacing: 12) {
                Label("Active phase age", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(activityColor)
                Label("Prefill completed", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue)
                Label("Session build completed", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.purple)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var cacheReuseChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prefill Reuse (/sec)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart {
                ForEach(stats.cacheReuseHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Tokens", point.value)
                    )
                    .foregroundStyle(.green.opacity(0.75))
                }
                ForEach(stats.cacheRebuildHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Tokens", point.value)
                    )
                    .foregroundStyle(.red.opacity(0.65))
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.0f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .frame(height: 150)

            HStack(spacing: 12) {
                Label("Reused", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                Label("Rebuilt", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var cacheFootprintChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Cache Footprint (est)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart(stats.cacheFootprintHistory) { point in
                LineMark(
                    x: .value("Time", point.timestamp),
                    y: .value("MB", point.value / 1_048_576)
                )
                .foregroundStyle(.orange)
                .interpolationMethod(.monotone)

                AreaMark(
                    x: .value("Time", point.timestamp),
                    y: .value("MB", point.value / 1_048_576)
                )
                .foregroundStyle(.orange.opacity(0.12))
                .interpolationMethod(.monotone)
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.1f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .frame(height: 150)
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var cacheSessionChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Cached Sessions")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart {
                ForEach(stats.cacheEntryHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Cached", point.value)
                    )
                    .foregroundStyle(.purple)
                    .interpolationMethod(.monotone)
                }
                ForEach(stats.activeSessionHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Active", point.value)
                    )
                    .foregroundStyle(.blue)
                    .interpolationMethod(.monotone)
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.0f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .frame(height: 150)

            HStack(spacing: 12) {
                Label("Cached", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.purple)
                Label("Active", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private var maxTokenRate: Double {
        stats.tokenRateHistory.map(\.value).max() ?? 10
    }

    // MARK: - Token Throughput Chart

    @ViewBuilder
    private var tokenThroughputChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Token Throughput (/sec)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Chart {
                ForEach(stats.promptTokenHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Tokens", point.value)
                    )
                    .foregroundStyle(.blue.opacity(0.7))
                }
                ForEach(stats.generationTokenHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Tokens", point.value)
                    )
                    .foregroundStyle(.orange.opacity(0.7))
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .second, count: 30)) { _ in
                    AxisGridLine()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.0f", v))
                                .font(.caption2.monospacedDigit())
                        }
                    }
                }
            }
            .frame(height: 150)

            // Legend
            HStack(spacing: 12) {
                Label("Prompt", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue)
                Label("Generation", systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Context Gauge

    @ViewBuilder
    private var contextGauge: some View {
        VStack(spacing: 8) {
            Text("Context")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            let maxCtx = max(stats.contextMax, modelManager.currentModel?.contextLength ?? 0)
            let used = stats.contextUsed
            let ratio = maxCtx > 0 ? Double(used) / Double(maxCtx) : 0

            Gauge(value: ratio) {
                EmptyView()
            } currentValueLabel: {
                Text(formatTokenCount(used))
                    .font(.title3.monospacedDigit().bold())
            } minimumValueLabel: {
                Text("0")
                    .font(.caption2)
            } maximumValueLabel: {
                Text(formatTokenCount(maxCtx))
                    .font(.caption2)
            }
            .gaugeStyle(.accessoryCircular)
            .scaleEffect(1.3)
            .tint(contextGradient(ratio: ratio))

            Text("\(Int(ratio * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private func contextGradient(ratio: Double) -> Color {
        if ratio > 0.9 { return .red }
        if ratio > 0.7 { return .orange }
        return .blue
    }

    // MARK: - GPU Memory Gauge

    @ViewBuilder
    private var gpuMemoryGauge: some View {
        VStack(spacing: 8) {
            Text("GPU Memory")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            let activeMB = Double(MLX.GPU.activeMemory) / 1_048_576
            let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576

            Text(String(format: "%.0f MB", activeMB))
                .font(.title3.monospacedDigit().bold())

            if peakMB > 0 {
                Text(String(format: "Peak: %.0f MB", peakMB))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Requests Card

    @ViewBuilder
    private var requestsCard: some View {
        VStack(spacing: 8) {
            Text("Requests")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Text("\(stats.totalRequests)")
                .font(.title3.monospacedDigit().bold())

            if stats.activeRequests > 0 {
                Text("\(stats.activeRequests) active")
                    .font(.caption2)
                    .foregroundStyle(.green)
            } else {
                Text("none active")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var cacheCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Session Cache")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            Text("\(stats.cacheEntryCount)")
                .font(.title3.monospacedDigit().bold())

            LabeledContent("Warm") {
                Text("\(stats.warmCacheEntryCount)")
                    .monospacedDigit()
            }
            .font(.caption)

            LabeledContent("Active") {
                Text("\(stats.activeCacheEntryCount)")
                    .monospacedDigit()
            }
            .font(.caption)

            LabeledContent("Est. Footprint") {
                Text(formatByteCount(stats.cacheEstimatedBytes))
                    .monospacedDigit()
            }
            .font(.caption)

            LabeledContent("Cached Tokens") {
                Text(formatTokenCount(stats.cacheEstimatedTokens))
                    .monospacedDigit()
            }
            .font(.caption)

            LabeledContent("Hit Rate") {
                Text(String(format: "%.0f%%", cacheHitRate * 100))
                    .monospacedDigit()
            }
            .font(.caption)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Cumulative

    @ViewBuilder
    private var cumulativeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Cumulative")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            LazyVGrid(columns: cardColumns, alignment: .leading, spacing: 12) {
                statTile(title: "Prompt Tokens", value: formatTokenCount(stats.totalPromptTokens), color: .blue)
                statTile(title: "Generated Tokens", value: formatTokenCount(stats.totalGenerationTokens), color: .orange)
                statTile(title: "Cache Hits", value: "\(stats.totalCacheHits)", color: .green)
                statTile(title: "Cache Misses", value: "\(stats.totalCacheMisses)", color: .red)
                statTile(title: "Reused Prefill", value: formatTokenCount(stats.totalCacheReusePromptTokens), color: .green)
                statTile(title: "Rebuilt Prefill", value: formatTokenCount(stats.totalCacheRebuildPromptTokens), color: .red)
                statTile(title: "Evictions", value: "\(stats.totalCacheEvictions)", color: .secondary)
                statTile(title: "Total Tokens", value: formatTokenCount(stats.totalPromptTokens + stats.totalGenerationTokens), color: .primary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var sessionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Cached Chat Sessions")
                    .font(.headline)
                Spacer()
                Text("\(stats.cachedSessions.count) visible")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if stats.cachedSessions.isEmpty {
                Text("No cached sessions yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ForEach(stats.cachedSessions) { session in
                    sessionRow(session)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Helpers

    @ViewBuilder
    private func phaseChip(title: String, count: Int, color: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(title)
            Text("\(count)")
                .monospacedDigit()
        }
        .font(.caption)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.12), in: Capsule())
    }

    @ViewBuilder
    private func statTile(title: String, value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.monospacedDigit().bold())
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func sessionRow(_ session: ConversationSessionCache.SessionSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(color(for: session.phase))
                        .frame(width: 8, height: 8)
                    Text(session.modelId)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                }
                Spacer()
                Text(session.phase.rawValue)
                    .font(.caption.monospacedDigit())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(color(for: session.phase).opacity(0.14), in: Capsule())
            }

            HStack(spacing: 12) {
                sessionMetric("Msgs", "\(session.messageCount)")
                sessionMetric("Cached", formatTokenCount(session.cachedTokenEstimate))
                sessionMetric("Reuse", formatTokenCount(session.lastReuseTokens))
                sessionMetric("Footprint", formatByteCount(session.estimatedBytes))
                sessionMetric("Hits", "\(session.hitCount)")
                sessionMetric("Active", "\(session.inFlightRequests)")
            }

            HStack(spacing: 12) {
                sessionMetric("Prompt", formatTokenCount(session.lastPromptTokens))
                sessionMetric("Completion", formatTokenCount(session.lastCompletionTokens))
                sessionMetric("Last Access", relativeTimeString(session.lastAccessAt))
            }

            let ratio = maxContextRatio(for: session.cachedTokenEstimate)
            ProgressView(value: ratio) {
                Text("Cached Context")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } currentValueLabel: {
                Text("\(Int(ratio * 100))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .tint(color(for: session.phase))
        }
        .padding(12)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private func sessionMetric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit().bold())
        }
    }

    private func formatTokenCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fk", Double(count) / 1_000)
        }
        return "\(count)"
    }

    private func formatByteCount(_ count: Int) -> String {
        let bytes = Double(count)
        if bytes >= 1_048_576 {
            return String(format: "%.1f MB", bytes / 1_048_576)
        }
        if bytes >= 1024 {
            return String(format: "%.0f KB", bytes / 1024)
        }
        return "\(count) B"
    }

    private func relativeTimeString(_ date: Date) -> String {
        let seconds = max(0, Int(Date.now.timeIntervalSince(date)))
        if seconds < 60 {
            return "\(seconds)s"
        }
        let minutes = seconds / 60
        if minutes < 60 {
            return "\(minutes)m"
        }
        return "\(minutes / 60)h"
    }

    private func color(for phase: APISessionPhase) -> Color {
        switch phase {
        case .idle:
            return .secondary
        case .sessionBuild:
            return .purple
        case .prefilling:
            return .blue
        case .generating:
            return .green
        }
    }

    private var cacheHitRate: Double {
        let total = stats.totalCacheHits + stats.totalCacheMisses
        guard total > 0 else { return 0 }
        return Double(stats.totalCacheHits) / Double(total)
    }

    private var phaseAgeLabel: String {
        if stats.generatingRequests > 0 { return "Generating s" }
        if stats.prefillingRequests > 0 { return "Prefill s" }
        if stats.sessionBuildRequests > 0 { return "Build s" }
        return "Preparing s"
    }

    private func maxContextRatio(for tokens: Int) -> Double {
        let maxContext = max(stats.contextMax, modelManager.currentModel?.contextLength ?? 0)
        guard maxContext > 0 else { return 0 }
        return min(1, Double(tokens) / Double(maxContext))
    }
}
