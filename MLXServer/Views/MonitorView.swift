import Charts
import MLX
import SwiftUI

/// Real-time inference monitoring dashboard, shown in place of the chat UI.
struct MonitorView: View {
    let stats: InferenceStats
    @Environment(ModelManager.self) private var modelManager

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Live status header
                liveStatusSection

                // Charts
                HStack(alignment: .top, spacing: 16) {
                    tokenRateChart
                    tokenThroughputChart
                }

                // Gauges row
                HStack(spacing: 16) {
                    contextGauge
                    gpuMemoryGauge
                    requestsCard
                }

                // Cumulative stats
                cumulativeSection
            }
            .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }

    // MARK: - Live Status

    @ViewBuilder
    private var liveStatusSection: some View {
        HStack(spacing: 16) {
            // Activity indicator
            HStack(spacing: 8) {
                Circle()
                    .fill(activityColor)
                    .frame(width: 10, height: 10)
                    .overlay {
                        if stats.isGenerating || stats.isPrefilling {
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
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private var activityColor: Color {
        if stats.isPrefilling { return .blue }
        if stats.isGenerating { return .green }
        if stats.activeRequests > 0 { return .orange }
        return .secondary
    }

    private var activityLabel: String {
        if stats.isPrefilling { return "Prefilling" }
        if stats.isGenerating { return "Generating" }
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

    // MARK: - Cumulative

    @ViewBuilder
    private var cumulativeSection: some View {
        HStack(spacing: 24) {
            VStack(spacing: 2) {
                Text("Total Prompt Tokens")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(formatTokenCount(stats.totalPromptTokens))
                    .font(.callout.monospacedDigit().bold())
                    .foregroundStyle(.blue)
            }

            VStack(spacing: 2) {
                Text("Total Generated Tokens")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(formatTokenCount(stats.totalGenerationTokens))
                    .font(.callout.monospacedDigit().bold())
                    .foregroundStyle(.orange)
            }

            VStack(spacing: 2) {
                Text("Total Tokens")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(formatTokenCount(stats.totalPromptTokens + stats.totalGenerationTokens))
                    .font(.callout.monospacedDigit().bold())
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Helpers

    private func formatTokenCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fk", Double(count) / 1_000)
        }
        return "\(count)"
    }
}
