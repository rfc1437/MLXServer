import SwiftUI

/// Modal overlay shown when a model is being downloaded from HuggingFace.
struct DownloadModalView: View {
    @Environment(ModelManager.self) private var modelManager

    var body: some View {
        VStack(spacing: 20) {
            // Header
            Label("Downloading Model", systemImage: "arrow.down.circle")
                .font(.headline)

            Text(modelManager.loadingModelName)
                .font(.title3.weight(.medium))
                .foregroundStyle(.primary)

            // Progress bar
            VStack(spacing: 8) {
                ProgressView(value: modelManager.downloadProgress)
                    .progressViewStyle(.linear)

                HStack {
                    // Files progress
                    if modelManager.downloadFilesTotal > 0 {
                        Text("File \(modelManager.downloadFilesCompleted)/\(modelManager.downloadFilesTotal)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    // Percentage
                    Text("\(Int(modelManager.downloadProgress * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                // Speed
                if modelManager.downloadSpeed > 0 {
                    Text(formatSpeed(modelManager.downloadSpeed))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }

            Text("The model will be cached locally for future use.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
        .frame(width: 380)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(radius: 20)
    }

    private func formatSpeed(_ bytesPerSec: Double) -> String {
        if bytesPerSec >= 1_073_741_824 {
            return String(format: "%.1f GB/s", bytesPerSec / 1_073_741_824)
        } else if bytesPerSec >= 1_048_576 {
            return String(format: "%.1f MB/s", bytesPerSec / 1_048_576)
        } else if bytesPerSec >= 1024 {
            return String(format: "%.0f KB/s", bytesPerSec / 1024)
        } else {
            return String(format: "%.0f B/s", bytesPerSec)
        }
    }
}
