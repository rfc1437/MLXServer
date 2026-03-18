import SwiftUI

struct ModelPickerView: View {
    @Environment(ModelManager.self) private var modelManager
    @State private var confirmRedownload: ModelConfig?

    var body: some View {
        HStack(spacing: 8) {
            Picker("Model", selection: selectedModelBinding) {
                ForEach(ModelConfig.availableModels) { config in
                    Label(
                        config.displayName,
                        systemImage: config.isLocal ? "checkmark.circle.fill" : "arrow.down.circle"
                    ).tag(config.id)
                }
            }
            .frame(width: 160)
            .disabled(modelManager.isLoading)

            // Re-download button (visible when a model is loaded)
            if let current = modelManager.currentModel, !modelManager.isLoading {
                Button {
                    confirmRedownload = current
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("Re-download \(current.displayName)")
            }
        }
        .alert("Re-download Model?", isPresented: .init(
            get: { confirmRedownload != nil },
            set: { if !$0 { confirmRedownload = nil } }
        )) {
            Button("Re-download", role: .destructive) {
                if let config = confirmRedownload {
                    Task { await modelManager.redownloadModel(config) }
                }
            }
            Button("Cancel", role: .cancel) {
                confirmRedownload = nil
            }
        } message: {
            if let config = confirmRedownload {
                Text("This will delete the local cache for \(config.displayName) and download it again from HuggingFace.")
            }
        }
    }

    private var selectedModelBinding: Binding<String> {
        Binding(
            get: { modelManager.currentModel?.id ?? ModelConfig.default.id },
            set: { newId in
                guard let config = ModelConfig.availableModels.first(where: { $0.id == newId }) else { return }
                Task {
                    await modelManager.loadModel(config)
                }
            }
        )
    }
}
