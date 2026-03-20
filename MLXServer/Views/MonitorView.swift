import Charts
import MLX
import SwiftUI

/// Real-time system dashboard focused on the stateless API server and prefix cache.
struct MonitorView: View {
    let stats: InferenceStats
    @Environment(ModelManager.self) private var modelManager

    private let chartColumns = [
        GridItem(.flexible(minimum: 280), spacing: 16),
        GridItem(.flexible(minimum: 280), spacing: 16),
    ]

    private let metricColumns = [
        GridItem(.flexible(minimum: 180), spacing: 16),
        GridItem(.flexible(minimum: 180), spacing: 16),
        GridItem(.flexible(minimum: 180), spacing: 16),
    ]

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                systemHeader

                LazyVGrid(columns: metricColumns, alignment: .leading, spacing: 16) {
                    metricCard(
                        title: "Requests",
                        value: "\(stats.totalRequests)",
                        detail: stats.activeRequests > 0 ? "\(stats.activeRequests) active" : "idle",
                        color: stats.activeRequests > 0 ? .green : .secondary
                    )
                    metricCard(
                        title: "Cache Entries",
                        value: "\(stats.cacheEntryCount)",
                        detail: formatTokenCount(stats.cacheEstimatedTokens) + " cached tokens",
                        color: .orange
                    )
                    metricCard(
                        title: "Cache Hit Rate",
                        value: String(format: "%.0f%%", stats.cacheHitRatePercent),
                        detail: "\(stats.totalCacheHits) hits / \(stats.totalCacheMisses) misses",
                        color: .blue
                    )
                    metricCard(
                        title: "Prefill Reuse",
                        value: formatTokenCount(stats.totalCacheReusePromptTokens),
                        detail: stats.currentCacheMatchedPromptTokens > 0
                            ? String(format: "%.0f%% match now", stats.currentCacheMatchQualityPercent)
                            : String(format: "%.0f%% total quality", stats.totalCacheMatchQualityPercent),
                        color: .teal
                    )
                    metricCard(
                        title: "Context",
                        value: formatTokenCount(stats.contextUsed),
                        detail: ofTotalContext,
                        color: contextColor
                    )
                    metricCard(
                        title: "GPU Memory",
                        value: formatByteCount(Int(MLX.GPU.activeMemory)),
                        detail: "peak " + formatByteCount(Int(MLX.GPU.peakMemory)),
                        color: .purple
                    )
                    metricCard(
                        title: "Generation Speed",
                        value: stats.isGenerating ? String(format: "%.1f tok/s", stats.currentTokensPerSecond) : "0 tok/s",
                        detail: "\(stats.currentGenerationTokens) output tokens",
                        color: .green
                    )
                }

                LazyVGrid(columns: chartColumns, alignment: .leading, spacing: 16) {
                    throughputChart
                    phaseChart
                    cacheChart
                    memoryChart
                }

                cumulativeSection
                cacheEntriesSection
            }
            .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }

    private var systemHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Circle()
                    .fill(systemStateColor)
                    .frame(width: 12, height: 12)
                Text(systemStateLabel)
                    .font(.headline)
                Spacer()
                if stats.activeRequests > 0 {
                    Text("phase age " + String(format: "%.0fs", stats.currentPhaseElapsed))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 8) {
                statusChip(title: "Preparing", value: stats.preparingRequests, color: .secondary)
                statusChip(title: "Prefill", value: stats.prefillingRequests, color: .blue)
                statusChip(title: "Generating", value: stats.generatingRequests, color: .green)
                statusChip(title: "Cache", value: stats.cacheEntryCount, color: .orange)
                statusChip(title: "Evictions", value: stats.totalCacheEvictions, color: .red)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var throughputChart: some View {
        chartCard(title: "Token Throughput") {
            Chart {
                ForEach(stats.promptTokenHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Prompt", point.value)
                    )
                    .foregroundStyle(.blue.opacity(0.7))
                }
                ForEach(stats.generationTokenHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Generation", point.value)
                    )
                    .foregroundStyle(.green.opacity(0.7))
                }
            }
            .chartXAxis { timeAxis }
            .chartYAxis { leadingValueAxis }
            .frame(height: 180)
        } footer: {
            legendRow(items: [("Prompt", .blue), ("Generation", .green)])
        }
    }

    private var phaseChart: some View {
        chartCard(title: "Phase Timing") {
            Chart {
                ForEach(stats.currentPhaseElapsedHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Age", point.value)
                    )
                    .foregroundStyle(systemStateColor)
                    .interpolationMethod(.monotone)
                }
                ForEach(stats.prefillDurationHistory) { point in
                    BarMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Prefill", point.value)
                    )
                    .foregroundStyle(.blue.opacity(0.45))
                }
            }
            .chartXAxis { timeAxis }
            .chartYAxis { leadingValueAxis }
            .frame(height: 180)
        } footer: {
            legendRow(items: [("Active phase age", systemStateColor), ("Prefill completions", .blue)])
        }
    }

    private var cacheChart: some View {
        chartCard(title: "Cache Match Quality") {
            Chart {
                ForEach(stats.cacheMatchQualityHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Match Quality", point.value)
                    )
                    .foregroundStyle(.teal)
                    .interpolationMethod(.monotone)
                }
                ForEach(stats.cacheHitRateHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Hit Rate", point.value)
                    )
                    .foregroundStyle(.blue)
                    .interpolationMethod(.monotone)
                }
            }
            .chartXAxis { timeAxis }
            .chartYAxis { leadingValueAxis }
            .frame(height: 180)
        } footer: {
            legendRow(items: [("Match quality %", .teal), ("Hit rate %", .blue)])
        }
    }

    private var memoryChart: some View {
        chartCard(title: "Estimated Cache Memory") {
            Chart {
                ForEach(stats.cacheFootprintHistory) { point in
                    AreaMark(
                        x: .value("Time", point.timestamp),
                        y: .value("MB", point.value / 1_048_576)
                    )
                    .foregroundStyle(.orange.opacity(0.15))
                    .interpolationMethod(.monotone)

                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("MB", point.value / 1_048_576)
                    )
                    .foregroundStyle(.orange)
                    .interpolationMethod(.monotone)
                }
                ForEach(stats.cacheMemoryPressureHistory) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Pressure", point.value)
                    )
                    .foregroundStyle(.red)
                    .interpolationMethod(.monotone)
                }
            }
            .chartXAxis { timeAxis }
            .chartYAxis { leadingValueAxis }
            .frame(height: 180)
        } footer: {
            legendRow(items: [("Estimated MB", .orange), ("Estimated budget %", .red)])
        }
    }

    private var cumulativeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Totals")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            LazyVGrid(columns: metricColumns, alignment: .leading, spacing: 12) {
                compactTile(title: "Prompt Tokens", value: formatTokenCount(stats.totalPromptTokens), color: .blue)
                compactTile(title: "Generated Tokens", value: formatTokenCount(stats.totalGenerationTokens), color: .green)
                compactTile(title: "Cache Evictions", value: "\(stats.totalCacheEvictions)", color: .red)
                compactTile(title: "Reused Prefill", value: formatTokenCount(stats.totalCacheReusePromptTokens), color: .teal)
                compactTile(title: "Rebuilt Prefill", value: formatTokenCount(stats.totalCacheRebuildPromptTokens), color: .orange)
                compactTile(title: "Match Quality", value: String(format: "%.0f%%", stats.totalCacheMatchQualityPercent), color: .teal)
                compactTile(title: "Prefill Time", value: String(format: "%.1fs", stats.totalPrefillDuration), color: .blue)
                compactTile(title: "Generation Time", value: String(format: "%.1fs", stats.totalGenerationDuration), color: .green)
                compactTile(title: "Cache Budget", value: formatByteCount(stats.cacheMemoryBudgetBytes), color: .orange)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var cacheEntriesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Prefix Cache Entries")
                    .font(.headline)
                Spacer()
                Text("\(stats.cachedEntries.count) visible")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if stats.cachedEntries.isEmpty {
                Text("No cache entries stored yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(stats.cachedEntries) { entry in
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text(entry.modelId)
                                .font(.callout.weight(.semibold))
                                .lineLimit(1)
                            Spacer()
                            Text(relativeTimeString(entry.lastAccessAt))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }

                        HStack(spacing: 16) {
                            entryMetric("Tokens", formatTokenCount(entry.tokenCount))
                            entryMetric("Est. Footprint", formatByteCount(entry.estimatedBytes))
                            entryMetric("Hits", "\(entry.hitCount)")
                            entryMetric("Created", relativeTimeString(entry.createdAt))
                        }
                    }
                    .padding(12)
                    .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func metricCard(title: String, value: String, detail: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.monospacedDigit().bold())
                .foregroundStyle(color)
            Text(detail)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func compactTile(title: String, value: String, color: Color) -> some View {
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

    private func chartCard<Content: View, Footer: View>(title: String, @ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            content()
            footer()
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func statusChip(title: String, value: Int, color: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(title)
            Text("\(value)")
                .monospacedDigit()
        }
        .font(.caption)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.12), in: Capsule())
    }

    private func entryMetric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit().bold())
        }
    }

    private func legendRow(items: [(String, Color)]) -> some View {
        HStack(spacing: 12) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                Label(item.0, systemImage: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(item.1)
            }
        }
    }

    private var timeAxis: some AxisContent {
        AxisMarks(values: .stride(by: .second, count: 30)) { _ in
            AxisGridLine()
        }
    }

    private var leadingValueAxis: some AxisContent {
        AxisMarks(position: .leading) { value in
            AxisGridLine()
            AxisValueLabel {
                if let doubleValue = value.as(Double.self) {
                    Text(String(format: "%.0f", doubleValue))
                        .font(.caption2.monospacedDigit())
                }
            }
        }
    }

    private var systemStateColor: Color {
        if stats.generatingRequests > 0 { return .green }
        if stats.prefillingRequests > 0 { return .blue }
        if stats.preparingRequests > 0 { return .orange }
        return .secondary
    }

    private var systemStateLabel: String {
        if stats.generatingRequests > 0 { return "Generating" }
        if stats.prefillingRequests > 0 { return "Prefilling" }
        if stats.preparingRequests > 0 { return "Preparing" }
        return "Idle"
    }

    private var contextColor: Color {
        let maxContext = max(stats.contextMax, modelManager.currentModel?.contextLength ?? 0)
        guard maxContext > 0 else { return .secondary }
        let ratio = Double(stats.contextUsed) / Double(maxContext)
        if ratio > 0.9 { return .red }
        if ratio > 0.7 { return .orange }
        return .blue
    }

    private var ofTotalContext: String {
        let total = max(stats.contextMax, modelManager.currentModel?.contextLength ?? 0)
        guard total > 0 else { return "no model" }
        return "of " + formatTokenCount(total)
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
        if bytes >= 1_073_741_824 {
            return String(format: "%.2f GB", bytes / 1_073_741_824)
        }
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
        if seconds < 60 { return "\(seconds)s ago" }
        let minutes = seconds / 60
        if minutes < 60 { return "\(minutes)m ago" }
        return "\(minutes / 60)h ago"
    }
}
