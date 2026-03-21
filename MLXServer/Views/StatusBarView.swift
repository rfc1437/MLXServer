import MLX
import SwiftUI

struct StatusBarView: View {
    let viewModel: ChatViewModel
    @Environment(ModelManager.self) private var modelManager

    var body: some View {
        HStack(spacing: 16) {
            // Model info
            if modelManager.isLoading {
                let pct = Int(modelManager.downloadProgress * 100)
                Text("Loading \(modelManager.loadingModelName)… \(pct)%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.orange)
            } else if let model = modelManager.currentModel {
                Label(model.displayName, systemImage: "cpu")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Text("\(model.contextLength / 1000)k ctx")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Text("No model loaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Label(viewModel.activeSceneName, systemImage: "theatermasks")
                .font(.caption)
                .foregroundStyle(.secondary)

            if let model = modelManager.currentModel, model.contextLength > 0 {
                contextFillView(totalContext: model.contextLength)
            }

            Spacer()

            // GPU memory
            let activeMB = Double(MLX.GPU.activeMemory) / 1_048_576
            if activeMB > 0 {
                Text(String(format: "GPU: %.0f MB", activeMB))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }

            // Token generation speed
            if viewModel.isGenerating {
                Text(String(format: "%.1f tok/s", viewModel.tokensPerSecond))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            // Token counts
            if viewModel.promptTokens > 0 || viewModel.generationTokens > 0 {
                Text("\(viewModel.promptTokens)→\(viewModel.generationTokens) tok")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }

            // API server status
            if viewModel.apiServer.isRunning {
                Label("API :\(viewModel.apiServer.port)", systemImage: "network")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Label("API off", systemImage: "network.slash")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            // Error
            if let error = modelManager.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .background(.bar)
    }

    @ViewBuilder
    private func contextFillView(totalContext: Int) -> some View {
        let usedTokens = viewModel.contextUsedTokens
        let ratio = viewModel.contextFillRatio
        let percent = Int((ratio * 100).rounded())

        HStack(spacing: 6) {
            Capsule()
                .fill(.quaternary)
                .frame(width: 48, height: 6)
                .overlay(alignment: .leading) {
                    Capsule()
                        .fill(contextFillColor(for: ratio))
                        .frame(width: max(4, 48 * ratio), height: 6)
                }

            Text("Ctx \(percent)%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .help("Approximate context usage: \(formatTokenCount(usedTokens)) of \(formatTokenCount(totalContext)) tokens")
    }

    private func contextFillColor(for ratio: Double) -> Color {
        if ratio >= 0.9 { return .red }
        if ratio >= 0.7 { return .orange }
        return .blue
    }

    private func formatTokenCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        }
        if count >= 1_000 {
            return String(format: "%.1fk", Double(count) / 1_000)
        }
        return "\(count)"
    }
}
