import SwiftUI

struct ModelPickerView: View {
    @Environment(ModelManager.self) private var modelManager

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
