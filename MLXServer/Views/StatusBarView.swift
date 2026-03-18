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
}
